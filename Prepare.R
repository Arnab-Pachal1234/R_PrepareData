#' @importFrom caret createDataPartition
#' @importFrom stats median model.matrix sd setNames
#' @export
#' 
library(caret)
library(class)
library(nnet)
library(rpart)
library(e1071)
library(randomForest)
library(glmnet)

prepare_data <- function(data,
                         target = names(data)[ncol(data)],
                         scale_type = "standard",
                         num_method = "mean",
                         p = 0.8,
                         seed = 123) {

  # ==========================================
  # 1. VALIDATION AND INITIAL CHECKS
  # ==========================================

  if (!is.data.frame(data) || nrow(data) == 0) {
    stop("Provide a non-empty data frame.")
  }

  if (anyDuplicated(names(data))) {
    stop("Duplicate column names found.")
  }

  if (!target %in% names(data)) {
    stop("Target column not found.")
  }

  scale_type <- tolower(scale_type)
  num_method <- tolower(num_method)

  if (!scale_type %in% c("standard", "minmax")) {
    stop("Use standard or minmax scaling.")
  }

  if (!num_method %in% c("mean", "median")) {
    stop("Use mean or median imputation.")
  }

  if (p <= 0 || p >= 1) {
    stop("p must be between 0 and 1.")
  }

  cat("\n========== DATASET INFORMATION ==========\n")

  cat("Original dimensions:", dim(data), "\n")
  cat("Target variable:", target, "\n")

  # Work on a copy
  data <- as.data.frame(data)

  # ==========================================
  # 2. CLEAN EMPTY STRINGS
  # ==========================================

  for (col in names(data)) {

    if (is.character(data[[col]]) ||
        is.factor(data[[col]])) {

      x <- trimws(as.character(data[[col]]))

      x[x == ""] <- NA

      data[[col]] <- x
    }
  }

  # ==========================================
  # 3. CHECK AND REMOVE DUPLICATES
  # ==========================================

  duplicate_count <- sum(duplicated(data))

  cat("\nDuplicate rows found:", duplicate_count, "\n")

  data <- data[!duplicated(data), , drop = FALSE]

  cat("Dimensions after duplicate removal:",
      dim(data), "\n")

  # ==========================================
  # 4. HANDLE MISSING TARGET VALUES
  # ==========================================

  missing_target <- sum(is.na(data[[target]]))

  cat("\nMissing target values:", missing_target, "\n")

  if (missing_target > 0) {

    cat("Removing rows with missing target values.\n")

    data <- data[!is.na(data[[target]]), , drop = FALSE]
  }

  if (nrow(data) < 2) {
    stop("Not enough observations after cleaning.")
  }

  # Classification target
  if (is.character(data[[target]])) {
    data[[target]] <- factor(data[[target]])
  }

  if (is.factor(data[[target]]) &&
      nlevels(droplevels(data[[target]])) < 2) {
    stop("Target must contain at least two classes.")
  }

  # ==========================================
  # 5. SEPARATE X AND Y
  # ==========================================

  X <- data[, setdiff(names(data), target),
            drop = FALSE]

  Y <- data[[target]]

  if (ncol(X) == 0) {
    stop("No predictor columns available.")
  }

  # Remove unused factor levels
  for (col in names(X)) {
    if (is.factor(X[[col]])) {
      X[[col]] <- droplevels(X[[col]])
    }
  }

  # ==========================================
  # 6. TRAIN-TEST SPLIT
  # ==========================================

  set.seed(seed)

  train_index <- caret::createDataPartition(
    Y,
    p = p,
    list = FALSE
  )

  X_train <- X[train_index, , drop = FALSE]
  X_test  <- X[-train_index, , drop = FALSE]

  train_y <- Y[train_index]
  test_y  <- Y[-train_index]

  cat("\n========== TRAIN TEST SPLIT ==========\n")

  cat("Training rows:", nrow(X_train), "\n")
  cat("Testing rows:", nrow(X_test), "\n")

  # ==========================================
  # 7. MISSING VALUE IMPUTATION
  # ==========================================

  imputation_info <- list()

  mode_value <- function(x) {

    x <- x[!is.na(x)]

    if (length(x) == 0) {
      stop("A categorical column contains only missing values.")
    }

    values <- unique(x)

    values[which.max(tabulate(match(x, values)))]
  }

  for (col in names(X_train)) {

    if (is.numeric(X_train[[col]])) {

      # Numerical: mean or median
      fill_value <- if (num_method == "mean") {
        mean(X_train[[col]], na.rm = TRUE)
      } else {
        median(X_train[[col]], na.rm = TRUE)
      }

      if (!is.finite(fill_value)) {
        stop(paste("Invalid imputation value in", col))
      }

      X_train[[col]][is.na(X_train[[col]])] <- fill_value
      X_test[[col]][is.na(X_test[[col]])] <- fill_value

      imputation_info[[col]] <- fill_value

    } else {

      # Categorical: mode
      fill_value <- mode_value(X_train[[col]])

      X_train[[col]][is.na(X_train[[col]])] <- fill_value
      X_test[[col]][is.na(X_test[[col]])] <- fill_value

      imputation_info[[col]] <- fill_value
    }
  }

  cat("\n========== MISSING VALUE IMPUTATION ==========\n")
  print(imputation_info)

  # ==========================================
  # 8. CATEGORICAL ENCODING
  # ==========================================

  # Convert categorical columns into factors
  for (col in names(X_train)) {

    if (!is.numeric(X_train[[col]])) {

      all_levels <- unique(c(
        as.character(X_train[[col]]),
        as.character(X_test[[col]])
      ))

      X_train[[col]] <- factor(
        X_train[[col]],
        levels = all_levels
      )

      X_test[[col]] <- factor(
        X_test[[col]],
        levels = all_levels
      )
    }
  }

  # One-hot encoding
  scale_train_x <- model.matrix(
    ~ . - 1,
    data = X_train
  )

  scale_test_x <- model.matrix(
    ~ . - 1,
    data = X_test
  )

  # Align test columns with training columns
  missing_columns <- setdiff(
    colnames(scale_train_x),
    colnames(scale_test_x)
  )

  for (col in missing_columns) {
    scale_test_x <- cbind(
      scale_test_x,
      setNames(
        data.frame(rep(0, nrow(scale_test_x))),
        col
      )
    )
  }

  scale_test_x <- scale_test_x[
    , colnames(scale_train_x), drop = FALSE
  ]

  # ==========================================
  # 9. FEATURE SCALING
  # ==========================================

  if (scale_type == "standard") {

    center <- colMeans(scale_train_x)

    spread <- apply(scale_train_x, 2, sd)

    spread[spread == 0 | is.na(spread)] <- 1

    scale_train_x <- scale(
      scale_train_x,
      center = center,
      scale = spread
    )

    scale_test_x <- scale(
      scale_test_x,
      center = center,
      scale = spread
    )

  } else {

    min_values <- apply(scale_train_x, 2, min)

    max_values <- apply(scale_train_x, 2, max)

    ranges <- max_values - min_values

    ranges[ranges == 0] <- 1

    scale_train_x <- sweep(
      sweep(scale_train_x, 2, min_values, "-"),
      2, ranges, "/"
    )

    scale_test_x <- sweep(
      sweep(scale_test_x, 2, min_values, "-"),
      2, ranges, "/"
    )
  }

  # ==========================================
  # 10. FINAL REPORT
  # ==========================================

  cat("\n========== FINAL REPORT ==========\n")

  cat("Training feature dimensions:",
      dim(scale_train_x), "\n")

  cat("Testing feature dimensions:",
      dim(scale_test_x), "\n")

  cat("Remaining training missing values:",
      sum(is.na(scale_train_x)), "\n")

  cat("Remaining testing missing values:",
      sum(is.na(scale_test_x)), "\n")

  # Return everything
  return(list(
    scale_train_x = scale_train_x,
    scale_test_x = scale_test_x,
    train_y = train_y,
    test_y = test_y,
    imputation_info = imputation_info,
    duplicate_count = duplicate_count
  ))
}



classification_model <- function(scale_train_x,
                                 scale_test_x,
                                 train_y,
                                 test_y,
                                 model = "knn",
                                 k = 5) {

  # ==========================================
  # 1. VALIDATE INPUTS
  # ==========================================

  model <- tolower(model)

  valid_models <- c(
    "logistic",
    "knn",
    "decision_tree",
    "naive_bayes",
    "svm",
    "random_forest"
  )

  if (!model %in% valid_models) {
    stop(paste(
      "Choose model from:",
      paste(valid_models, collapse = ", ")
    ))
  }

  # Check dimensions
  if (nrow(scale_train_x) != length(train_y) ||
      nrow(scale_test_x) != length(test_y)) {
    stop("Feature rows and target lengths do not match.")
  }

  if (ncol(scale_train_x) != ncol(scale_test_x)) {
    stop("Training and testing features must have same columns.")
  }

  if (anyNA(scale_train_x) || anyNA(scale_test_x)) {
    stop("Missing values found in features.")
  }

  if (length(unique(train_y)) < 2) {
    stop("Training target must contain at least two classes.")
  }

  if (k < 1 || k > length(train_y)) {
    stop("Invalid K value.")
  }

  # Make target categorical
  classes <- if (is.factor(train_y)) {
    levels(train_y)
  } else {
    unique(as.character(train_y))
  }

  train_y <- factor(train_y, levels = classes)
  test_y <- factor(test_y, levels = classes)

  # Convert features to numeric matrices
  train_x <- as.matrix(scale_train_x)
  test_x <- as.matrix(scale_test_x)

  storage.mode(train_x) <- "double"
  storage.mode(test_x) <- "double"

  # Check infinite values
  if (any(!is.finite(train_x)) ||
      any(!is.finite(test_x))) {
    stop("Infinite or non-finite feature values found.")
  }

  # Convert to data frames for formula-based models
  train_df <- data.frame(
    train_x,
    check.names = FALSE
  )

  test_df <- data.frame(
    test_x,
    check.names = FALSE
  )

  train_df$target <- train_y

  # ==========================================
  # 2. TRAIN CLASSIFICATION MODEL
  # ==========================================

  cat("\n====================================\n")
  cat("MODEL:", toupper(model), "\n")
  cat("====================================\n")

  if (model == "knn") {

    # K-Nearest Neighbors

    fitted_model <- NULL

    predictions <- class::knn(
      train = train_x,
      test = test_x,
      cl = train_y,
      k = k
    )

  } else if (model == "logistic") {

    # Multinomial Logistic Regression

    fitted_model <- nnet::multinom(
      target ~ .,
      data = train_df,
      trace = FALSE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df,
      type = "class"
    )

  } else if (model == "decision_tree") {

    # Decision Tree

    fitted_model <- rpart::rpart(
      target ~ .,
      data = train_df,
      method = "class"
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df,
      type = "class"
    )

  } else if (model == "naive_bayes") {

    # Naive Bayes

    fitted_model <- e1071::naiveBayes(
      target ~ .,
      data = train_df
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df,
      type = "class"
    )

  } else if (model == "svm") {

    # Support Vector Machine

    fitted_model <- e1071::svm(
      target ~ .,
      data = train_df,
      kernel = "radial",
      scale = FALSE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "random_forest") {

    # Random Forest

    fitted_model <- randomForest::randomForest(
      target ~ .,
      data = train_df,
      ntree = 500,
      importance = TRUE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )
  }

  # ==========================================
  # 3. PREDICTION
  # ==========================================

  predictions <- factor(
    predictions,
    levels = classes
  )

  # ==========================================
  # 4. CONFUSION MATRIX
  # ==========================================

  cm <- table(
    Actual = test_y,
    Predicted = predictions
  )

  cat("\n========== CONFUSION MATRIX ==========\n")

  print(cm)

  # ==========================================
  # 5. CALCULATE METRICS
  # ==========================================

  TP <- diag(cm)

  actual_total <- rowSums(cm)
  predicted_total <- colSums(cm)

  precision <- ifelse(
    predicted_total > 0,
    TP / predicted_total,
    0
  )

  recall <- ifelse(
    actual_total > 0,
    TP / actual_total,
    0
  )

  f1 <- ifelse(
    precision + recall > 0,
    2 * precision * recall /
      (precision + recall),
    0
  )

  # Class-wise metrics
  class_metrics <- data.frame(
    Class = rownames(cm),
    Precision = as.numeric(precision),
    Recall = as.numeric(recall),
    F1_Score = as.numeric(f1),
    Support = as.numeric(actual_total)
  )

  # Overall accuracy
  accuracy <- sum(TP) / sum(cm)

  # Macro averages
  macro_precision <- mean(precision)
  macro_recall <- mean(recall)
  macro_f1 <- mean(f1)

  # Weighted averages
  weighted_precision <- sum(precision * actual_total) /
    sum(actual_total)

  weighted_recall <- sum(recall * actual_total) /
    sum(actual_total)

  weighted_f1 <- sum(f1 * actual_total) /
    sum(actual_total)

  overall_metrics <- data.frame(
    Metric = c(
      "Accuracy",
      "Macro Precision",
      "Macro Recall",
      "Macro F1 Score",
      "Weighted Precision",
      "Weighted Recall",
      "Weighted F1 Score"
    ),

    Score = c(
      accuracy,
      macro_precision,
      macro_recall,
      macro_f1,
      weighted_precision,
      weighted_recall,
      weighted_f1
    )
  )

  # ==========================================
  # 6. DISPLAY RESULTS
  # ==========================================

  cat("\n========== CLASS-WISE METRICS ==========\n")

  print(class_metrics, row.names = FALSE)

  cat("\n========== OVERALL METRICS ==========\n")

  print(overall_metrics, row.names = FALSE)

  cat("\n========== MODEL SUMMARY ==========\n")

  cat("Model:", model, "\n")
  cat("Accuracy:", round(accuracy * 100, 2), "%\n")

  cat("Macro F1 Score:", round(macro_f1, 4), "\n")

  # ==========================================
  # 7. RETURN EVERYTHING
  # ==========================================

  return(list(

    model = fitted_model,

    predictions = predictions,

    confusion_matrix = cm,

    class_metrics = class_metrics,

    overall_metrics = overall_metrics,

    accuracy = accuracy

  ))
}



regression_model <- function(scale_train_x,
                             scale_test_x,
                             train_y,
                             test_y,
                             model = "multiple_linear",
                             degree = 2,
                             alpha = 0.5,
                             k = 5) {

  # ==========================================
  # 1. VALIDATE INPUTS
  # ==========================================

  model <- tolower(model)

  valid_models <- c(
    "linear",
    "multiple_linear",
    "polynomial",
    "lasso",
    "ridge",
    "elastic_net",
    "decision_tree",
    "random_forest",
    "svm"
  )

  if (!model %in% valid_models) {
    stop(paste(
      "Choose from:",
      paste(valid_models, collapse = ", ")
    ))
  }

  # Convert targets to numeric
  train_y <- as.numeric(train_y)
  test_y <- as.numeric(test_y)

  if (length(train_y) != nrow(scale_train_x) ||
      length(test_y) != nrow(scale_test_x)) {
    stop("Feature rows and target lengths do not match.")
  }

  if (nrow(scale_train_x) < 2 ||
      nrow(scale_test_x) < 1) {
    stop("Insufficient training or testing observations.")
  }

  if (anyNA(train_y) || anyNA(test_y) ||
      any(!is.finite(train_y)) ||
      any(!is.finite(test_y))) {
    stop("Target contains missing or non-finite values.")
  }

  if (anyNA(scale_train_x) || anyNA(scale_test_x)) {
    stop("Missing values found in features.")
  }

  if (degree < 1 || degree != as.integer(degree)) {
    stop("Polynomial degree must be a positive integer.")
  }

  if (alpha < 0 || alpha > 1) {
    stop("Alpha must be between 0 and 1.")
  }

  if (model == "linear" && ncol(scale_train_x) != 1) {
    stop("Simple linear regression requires exactly one feature.")
  }

  if (model == "multiple_linear" && ncol(scale_train_x) < 2) {
    stop("Multiple linear regression requires at least two features.")
  }

  if (model %in% c("lasso", "ridge", "elastic_net") &&
      nrow(scale_train_x) <= k) {
    stop("Training rows must exceed the CV fold count.")
  }

  # Convert features to numeric matrices
  train_x <- as.matrix(scale_train_x)
  test_x <- as.matrix(scale_test_x)

  storage.mode(train_x) <- "double"
  storage.mode(test_x) <- "double"

  if (any(!is.finite(train_x)) ||
      any(!is.finite(test_x))) {
    stop("Features contain infinite or non-finite values.")
  }

  if (ncol(train_x) != ncol(test_x)) {
    stop("Train and test feature counts do not match.")
  }

  if (any(apply(train_x, 2, function(x) sd(x) == 0))) {
    warning("Some training features have zero variance.")
  }

  # Make safe feature names
  feature_names <- paste0("X", seq_len(ncol(train_x)))

  colnames(train_x) <- feature_names
  colnames(test_x) <- feature_names

  train_df <- data.frame(train_x, check.names = FALSE)
  test_df <- data.frame(test_x, check.names = FALSE)

  train_df$target <- train_y

  # ==========================================
  # 2. TRAIN MODEL
  # ==========================================

  set.seed(123)

  cat("\n====================================\n")
  cat("REGRESSION MODEL:", toupper(model), "\n")
  cat("====================================\n")

  fitted_model <- NULL

  if (model %in% c("linear", "multiple_linear")) {

    # Simple / Multiple Linear Regression

    fitted_model <- lm(
      target ~ .,
      data = train_df
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "polynomial") {

    # Polynomial Regression
    # Creates powers of each feature, without interactions

    make_poly <- function(x, degree) {

      result <- as.data.frame(x)

      for (j in seq_len(ncol(x))) {

        for (d in 2:degree) {

          result[[paste0("X", j, "_power_", d)]] <-
            x[, j]^d
        }
      }

      result
    }

    # Degree 1 terms are already present
    poly_train <- make_poly(train_x, degree)
    poly_test <- make_poly(test_x, degree)

    poly_train$target <- train_y

    fitted_model <- lm(
      target ~ .,
      data = poly_train
    )

    predictions <- predict(
      fitted_model,
      newdata = poly_test
    )

  } else if (model %in% c("lasso", "ridge", "elastic_net")) {

    # L1 / L2 / Elastic Net Regularization

    if (model == "lasso") {
      alpha_value <- 1
    } else if (model == "ridge") {
      alpha_value <- 0
    } else {
      alpha_value <- alpha
    }

    fitted_model <- glmnet::cv.glmnet(
      x = train_x,
      y = train_y,
      alpha = alpha_value,
      family = "gaussian",
      nfolds = k,
      standardize = FALSE
    )

    predictions <- predict(
      fitted_model,
      newx = test_x,
      s = "lambda.min"
    )

    predictions <- as.numeric(predictions)

  } else if (model == "decision_tree") {

    # Decision Tree Regression

    fitted_model <- rpart::rpart(
      target ~ .,
      data = train_df,
      method = "anova"
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "random_forest") {

    # Random Forest Regression

    fitted_model <- randomForest::randomForest(
      target ~ .,
      data = train_df,
      ntree = 500,
      importance = TRUE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_df
    )

  } else if (model == "svm") {

    # Support Vector Regression

    fitted_model <- e1071::svm(
      x = train_x,
      y = train_y,
      type = "eps-regression",
      kernel = "radial",
      scale = FALSE
    )

    predictions <- predict(
      fitted_model,
      newdata = test_x
    )
  }

  predictions <- as.numeric(predictions)

  if (length(predictions) != length(test_y) ||
      any(!is.finite(predictions))) {
    stop("Model returned invalid predictions.")
  }

  # ==========================================
  # 3. CALCULATE ERRORS
  # ==========================================

  residuals <- test_y - predictions

  n <- length(test_y)
  p <- ncol(train_x)

  MAE <- mean(abs(residuals))

  MSE <- mean(residuals^2)

  RMSE <- sqrt(MSE)

  # ==========================================
  # 4. R SQUARED
  # ==========================================

  SST <- sum((test_y - mean(test_y))^2)

  SSE <- sum(residuals^2)

  R2 <- if (SST == 0) {
    NA_real_
  } else {
    1 - SSE / SST
  }

  # Adjusted R2
  adjusted_R2 <- if (is.na(R2) || n <= p + 1) {
    NA_real_
  } else {
    1 - (1 - R2) * (n - 1) / (n - p - 1)
  }

  # ==========================================
  # 5. MAPE
  # ==========================================

  nonzero_indices <- which(test_y != 0)

  MAPE <- if (length(nonzero_indices) == 0) {
    NA_real_
  } else {
    mean(
      abs(
        residuals[nonzero_indices] /
          test_y[nonzero_indices]
      )
    ) * 100
  }

  # ==========================================
  # 6. EVALUATION TABLE
  # ==========================================

  metrics <- data.frame(
    Metric = c(
      "MAE",
      "MSE",
      "RMSE",
      "R2",
      "Adjusted R2",
      "MAPE (%)"
    ),

    Value = c(
      MAE,
      MSE,
      RMSE,
      R2,
      adjusted_R2,
      MAPE
    )
  )

  # ==========================================
  # 7. PREDICTION TABLE
  # ==========================================

  prediction_table <- data.frame(
    Actual = test_y,
    Predicted = predictions,
    Residual = residuals
  )

  # ==========================================
  # 8. DISPLAY RESULTS
  # ==========================================

  cat("\n========== REGRESSION METRICS ==========\n")

  print(metrics, row.names = FALSE)

  cat("\n========== PREDICTION DETAILS ==========\n")

  print(head(prediction_table, 10))

  cat("\n========== MODEL SUMMARY ==========\n")

  cat("Model:", model, "\n")
  cat("Training observations:", length(train_y), "\n")
  cat("Testing observations:", n, "\n")
  cat("Number of features:", p, "\n")

  # ==========================================
  # 9. RETURN EVERYTHING
  # ==========================================

  return(list(

    model = fitted_model,

    predictions = predictions,

    actual = test_y,

    residuals = residuals,

    prediction_table = prediction_table,

    metrics = metrics,

    MAE = MAE,

    MSE = MSE,

    RMSE = RMSE,

    R2 = R2,

    adjusted_R2 = adjusted_R2,

    MAPE = MAPE

  ))
}
