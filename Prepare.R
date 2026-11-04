#' @importFrom caret createDataPartition
#' @importFrom stats median model.matrix sd setNames
#' @export
install.packages("caret")
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
