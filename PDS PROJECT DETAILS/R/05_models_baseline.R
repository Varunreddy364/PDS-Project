# ==============================================================================
# 05_models_baseline.R - Implementation & Baseline Evaluation of 12 ML Algorithms
# Authors: Karepalli Varun Reddy (24BDS1126) & Madheshwaran P (24BDS1159)
# Course: Programming for Data Science, VIT Chennai
# ==============================================================================

if (file.exists("DA2_DA3/R/functions.R")) {
  base_dir <- "DA2_DA3"
} else if (file.exists("R/functions.R")) {
  base_dir <- "."
} else {
  base_dir <- "."
}

source(file.path(base_dir, "R/functions.R"))
ensure_packages()
library(doParallel)

cat("\n======================================================================\n")
cat("=== PART 2 - SECTION C: 12 MACHINE LEARNING ALGORITHMS (BASELINE) ===\n")
cat("======================================================================\n\n")

tab_dir <- file.path(base_dir, "results/tables")
mod_dir <- file.path(base_dir, "results/models")
dir.create(mod_dir, recursive = TRUE, showWarnings = FALSE)

# Load train and test partitions
train_df <- readRDS(file.path(tab_dir, "train_data.rds"))
test_df  <- readRDS(file.path(tab_dir, "test_data.rds"))

# Clean inputs: remove continuous target automation_risk_score
train_clean <- train_df[, !(colnames(train_df) %in% c("automation_risk_score"))]
test_clean  <- test_df[, !(colnames(test_df) %in% c("automation_risk_score"))]

y_test <- test_clean$risk_category

cat("--> Training Partition: ", nrow(train_clean), " instances, ", ncol(train_clean) - 1, " predictor features\n", sep = "")
cat("--> Test Partition:     ", nrow(test_clean), " instances\n", sep = "")
cat("--> Setup: 5-Fold Cross-Validation, 3 Repeats, classProbs = TRUE, multiClassSummary\n\n")

# Setup parallel processing (3 worker threads)
cl <- makeCluster(3)
registerDoParallel(cl)

# Unified trainControl across all models
set.seed(123)
ctrl_baseline <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

# Helper function to evaluate and save model
evaluate_model <- function(fit, model_name, train_time, test_data, y_true) {
  # CV metrics from resamples
  cv_res <- fit$results[which.max(fit$results$Accuracy), ]
  cv_acc <- cv_res$Accuracy
  cv_kap <- cv_res$Kappa
  cv_f1  <- if ("Mean_F1" %in% colnames(cv_res)) cv_res$Mean_F1 else NA
  
  # Predict on held-out test set
  test_preds <- predict(fit, newdata = test_data)
  test_probs <- predict(fit, newdata = test_data, type = "prob")
  
  test_metrics <- compute_multiclass_metrics(
    y_true = y_true,
    y_pred = test_preds,
    prob_mat = test_probs,
    train_time_sec = train_time
  )
  
  cat(sprintf("--> [%s] Train Time: %.2fs | CV Acc: %.4f | Test Acc: %.4f | Macro F1: %.4f | Macro AUC: %.4f\n",
              model_name, train_time, cv_acc, test_metrics$Accuracy, test_metrics$Macro_F1, test_metrics$Macro_AUC))
  
  # Save model artifact
  saveRDS(fit, file.path(mod_dir, paste0("baseline_", gsub("[^A-Za-z0-9]", "_", tolower(model_name)), ".rds")))
  
  data.frame(
    Model_ID = model_name,
    CV_Accuracy = round(cv_acc, 4),
    CV_Kappa = round(cv_kap, 4),
    CV_Macro_F1 = round(cv_f1, 4),
    test_metrics
  )
}

baseline_results <- list()

# ------------------------------------------------------------------------------
# MODEL 0: Majority Class Benchmark
# ------------------------------------------------------------------------------
cat("=== BENCHMARK: MAJORITY CLASS CLASSIFIER ===\n")
maj_class <- names(which.max(table(train_clean$risk_category)))
maj_preds <- factor(rep(maj_class, nrow(test_clean)), levels = levels(y_test))
maj_probs <- matrix(0, nrow = nrow(test_clean), ncol = length(levels(y_test)))
colnames(maj_probs) <- levels(y_test)
maj_probs[, maj_class] <- 1.0

maj_metrics <- compute_multiclass_metrics(y_test, maj_preds, maj_probs, 0.01)
baseline_results[["Majority_Class"]] <- data.frame(
  Model_ID = "Majority Class Benchmark",
  CV_Accuracy = round(max(prop.table(table(train_clean$risk_category))), 4),
  CV_Kappa = 0.0000,
  CV_Macro_F1 = round(maj_metrics$Macro_F1, 4),
  maj_metrics
)
cat(sprintf("--> [Majority Class] Test Acc: %.4f | Macro F1: %.4f | Kappa: %.4f\n\n",
            maj_metrics$Accuracy, maj_metrics$Macro_F1, maj_metrics$Kappa))

# ------------------------------------------------------------------------------
# MODEL 1: Multinomial Logistic Regression (multinom)
# ------------------------------------------------------------------------------
cat("=== MODEL 1: MULTINOMIAL LOGISTIC REGRESSION (nnet::multinom) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_multinom <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "multinom", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale"),
  trace = FALSE, 
  MaxNWts = 3000
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Multinomial_Logit"]] <- evaluate_model(fit_multinom, "Multinomial Logistic Regression", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 2: Elastic-Net Logistic Regression (glmnet)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 2: ELASTIC-NET LOGISTIC REGRESSION (glmnet) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_glmnet <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "glmnet", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale"),
  family = "multinomial"
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Elastic_Net"]] <- evaluate_model(fit_glmnet, "Elastic-Net Logistic Regression", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 3: Linear Discriminant Analysis (MASS::lda)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 3: LINEAR DISCRIMINANT ANALYSIS (MASS::lda) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_lda <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "lda", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale", "corr")
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["LDA"]] <- evaluate_model(fit_lda, "Linear Discriminant Analysis", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 4: Naive Bayes (naivebayes)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 4: NAIVE BAYES (naivebayes) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_nb <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "naive_bayes", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale")
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Naive_Bayes"]] <- evaluate_model(fit_nb, "Naive Bayes", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 5: K-Nearest Neighbors (kknn)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 5: K-NEAREST NEIGHBORS (kknn) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_kknn <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "kknn", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale")
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["KNN"]] <- evaluate_model(fit_kknn, "K-Nearest Neighbors", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 6: Decision Tree (rpart)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 6: DECISION TREE (rpart) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_rpart <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "rpart", 
  trControl = ctrl_baseline
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Decision_Tree"]] <- evaluate_model(fit_rpart, "Decision Tree (CART)", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 7: C5.0 Rule/Tree-Based Classifier (C50)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 7: C5.0 CLASSIFIER (C50) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_c50 <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "C5.0", 
  trControl = ctrl_baseline
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["C50"]] <- evaluate_model(fit_c50, "C5.0 Decision Trees", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 8: Bagged Classification Trees (ipred::treebag)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 8: BAGGED TREES (ipred::treebag) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_treebag <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "treebag", 
  trControl = ctrl_baseline
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Bagged_Trees"]] <- evaluate_model(fit_treebag, "Bagged Decision Trees", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 9: Random Forest (ranger)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 9: RANDOM FOREST (ranger) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_ranger <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "ranger", 
  trControl = ctrl_baseline,
  importance = "impurity",
  num.trees = 150
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Random_Forest"]] <- evaluate_model(fit_ranger, "Random Forest (ranger)", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 10: Gradient Boosting Machine (gbm)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 10: GRADIENT BOOSTING MACHINE (gbm) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_gbm <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "gbm", 
  trControl = ctrl_baseline,
  verbose = FALSE
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Gradient_Boosting"]] <- evaluate_model(fit_gbm, "Gradient Boosting Machine", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 11: Support Vector Machine Radial (kernlab::svmRadial)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 11: SUPPORT VECTOR MACHINE RADIAL (svmRadial) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_svm <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "svmRadial", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale")
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["SVM_Radial"]] <- evaluate_model(fit_svm, "Support Vector Machine (Radial)", t1, test_clean, y_test)

# ------------------------------------------------------------------------------
# MODEL 12: Neural Network / MLP (nnet)
# ------------------------------------------------------------------------------
cat("\n=== MODEL 12: NEURAL NETWORK / MLP (nnet) ===\n")
set.seed(123)
t0 <- Sys.time()
fit_nnet <- train(
  risk_category ~ ., 
  data = train_clean, 
  method = "nnet", 
  trControl = ctrl_baseline,
  preProcess = c("zv", "center", "scale"),
  trace = FALSE, 
  MaxNWts = 5000
)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
baseline_results[["Neural_Network"]] <- evaluate_model(fit_nnet, "Artificial Neural Network (MLP)", t1, test_clean, y_test)

# Stop cluster
stopCluster(cl)
registerDoSEQ()

# ------------------------------------------------------------------------------
# MASTER BASELINE COMPARISON TABLE
# ------------------------------------------------------------------------------
cat("\n======================================================================\n")
cat("=== MASTER BASELINE PERFORMANCE RANKING TABLE ===\n")
cat("======================================================================\n\n")

baseline_df <- do.call(rbind, baseline_results)
baseline_df <- baseline_df %>% arrange(desc(Accuracy))
print(baseline_df[, c("Model_ID", "CV_Accuracy", "Accuracy", "Balanced_Accuracy", "Kappa", "Macro_F1", "Macro_AUC", "Training_Time_Sec")])

write.csv(baseline_df, file.path(tab_dir, "baseline_models_performance.csv"), row.names = FALSE)
cat("\n--> Baseline performance table written to results/tables/baseline_models_performance.csv\n")

cat("\n=== BASELINE MODELS EVALUATION COMPLETE ===\n\n")
