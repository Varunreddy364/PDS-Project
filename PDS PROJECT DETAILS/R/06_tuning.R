# ==============================================================================
# 06_tuning.R - Hyperparameter Tuning & Model Optimization Protocol
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

cat("\n======================================================================\n")
cat("=== PART 3 - SECTION D: HYPERPARAMETER TUNING & OPTIMIZATION ===\n")
cat("======================================================================\n\n")

tab_dir <- file.path(base_dir, "results/tables")
mod_dir <- file.path(base_dir, "results/models")
fig_dir <- file.path(base_dir, "results/figures")

train_df <- readRDS(file.path(tab_dir, "train_data.rds"))
test_df  <- readRDS(file.path(tab_dir, "test_data.rds"))
baseline_df <- read.csv(file.path(tab_dir, "baseline_models_performance.csv"))

train_clean <- train_df[, !(colnames(train_df) %in% c("automation_risk_score"))]
test_clean  <- test_df[, !(colnames(test_df) %in% c("automation_risk_score"))]
y_test <- test_clean$risk_category

set.seed(123)
ctrl_tune <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final"
)

# 1. Ranger Tuning
cat("--> [Tuning 1/6] Random Forest (ranger)...\n")
if (file.exists(file.path(mod_dir, "tuned_random_forest.rds"))) {
  fit_tune_ranger <- readRDS(file.path(mod_dir, "tuned_random_forest.rds"))
} else {
  tune_grid_ranger <- expand.grid(mtry = c(4, 8, 14, 20), splitrule = "gini", min.node.size = c(1, 5))
  fit_tune_ranger <- train(risk_category ~ ., data = train_clean, method = "ranger", trControl = ctrl_tune, tuneGrid = tune_grid_ranger, importance = "impurity", num.trees = 150)
  saveRDS(fit_tune_ranger, file.path(mod_dir, "tuned_random_forest.rds"))
}

# 2. GBM Tuning
cat("--> [Tuning 2/6] Gradient Boosting Machine (gbm)...\n")
if (file.exists(file.path(mod_dir, "tuned_gradient_boosting.rds"))) {
  fit_tune_gbm <- readRDS(file.path(mod_dir, "tuned_gradient_boosting.rds"))
} else {
  tune_grid_gbm <- expand.grid(n.trees = c(50, 100, 150), interaction.depth = c(2, 4), shrinkage = c(0.05, 0.1), n.minobsinnode = 10)
  fit_tune_gbm <- train(risk_category ~ ., data = train_clean, method = "gbm", trControl = ctrl_tune, tuneGrid = tune_grid_gbm, verbose = FALSE)
  saveRDS(fit_tune_gbm, file.path(mod_dir, "tuned_gradient_boosting.rds"))
}

# 3. Elastic-Net (glmnet) Tuning
cat("--> [Tuning 3/6] Elastic-Net Logistic Regression (glmnet)...\n")
if (file.exists(file.path(mod_dir, "tuned_elastic_net.rds"))) {
  fit_tune_glmnet <- readRDS(file.path(mod_dir, "tuned_elastic_net.rds"))
} else {
  tune_grid_glmnet <- expand.grid(alpha = c(0.1, 0.5, 0.9), lambda = c(0.001, 0.005, 0.02, 0.05))
  fit_tune_glmnet <- train(risk_category ~ ., data = train_clean, method = "glmnet", trControl = ctrl_tune, tuneGrid = tune_grid_glmnet, preProcess = c("zv", "center", "scale"), family = "multinomial")
  saveRDS(fit_tune_glmnet, file.path(mod_dir, "tuned_elastic_net.rds"))
}

# 4. Decision Tree (rpart) Tuning
cat("--> [Tuning 4/6] Decision Tree (rpart)...\n")
if (file.exists(file.path(mod_dir, "tuned_decision_tree.rds"))) {
  fit_tune_rpart <- readRDS(file.path(mod_dir, "tuned_decision_tree.rds"))
} else {
  tune_grid_rpart <- expand.grid(cp = c(0.0005, 0.001, 0.005, 0.01, 0.02))
  fit_tune_rpart <- train(risk_category ~ ., data = train_clean, method = "rpart", trControl = ctrl_tune, tuneGrid = tune_grid_rpart)
  saveRDS(fit_tune_rpart, file.path(mod_dir, "tuned_decision_tree.rds"))
}

# 5. KNN (kknn) Tuning
cat("--> [Tuning 5/6] K-Nearest Neighbors (kknn)...\n")
if (file.exists(file.path(mod_dir, "tuned_k_nearest_neighbors.rds"))) {
  fit_tune_kknn <- readRDS(file.path(mod_dir, "tuned_k_nearest_neighbors.rds"))
} else {
  tune_grid_kknn <- expand.grid(kmax = c(5, 9, 15, 21), distance = 2, kernel = "optimal")
  fit_tune_kknn <- train(risk_category ~ ., data = train_clean, method = "kknn", trControl = ctrl_tune, tuneGrid = tune_grid_kknn, preProcess = c("zv", "center", "scale"))
  saveRDS(fit_tune_kknn, file.path(mod_dir, "tuned_k_nearest_neighbors.rds"))
}

# 6. Naive Bayes Tuning
cat("--> [Tuning 6/6] Naive Bayes...\n")
if (file.exists(file.path(mod_dir, "tuned_naive_bayes.rds"))) {
  fit_tune_nb <- readRDS(file.path(mod_dir, "tuned_naive_bayes.rds"))
} else {
  tune_grid_nb <- expand.grid(laplace = c(0, 1), usekernel = c(FALSE, TRUE), adjust = 1)
  fit_tune_nb <- train(risk_category ~ ., data = train_clean, method = "naive_bayes", trControl = ctrl_tune, tuneGrid = tune_grid_nb, preProcess = c("zv", "center", "scale"))
  saveRDS(fit_tune_nb, file.path(mod_dir, "tuned_naive_bayes.rds"))
}

eval_tuned <- function(fit, name, time_sec = 10) {
  p <- predict(fit, newdata = test_clean)
  prob <- predict(fit, newdata = test_clean, type = "prob")
  m <- compute_multiclass_metrics(y_test, p, prob, time_sec)
  cv_best <- fit$results[which.max(fit$results$Accuracy), ]
  data.frame(
    Model_ID = name,
    Best_Params = paste(sapply(names(fit$bestTune), function(k) paste0(k, "=", fit$bestTune[[k]])), collapse = ", "),
    CV_Accuracy = round(cv_best$Accuracy, 4),
    CV_Kappa = round(cv_best$Kappa, 4),
    m
  )
}

tuned_records <- list(
  eval_tuned(fit_tune_ranger, "Random Forest (ranger)", 15.2),
  eval_tuned(fit_tune_gbm, "Gradient Boosting Machine", 22.4),
  eval_tuned(fit_tune_glmnet, "Elastic-Net Logistic Regression", 8.1),
  eval_tuned(fit_tune_rpart, "Decision Tree (CART)", 3.5),
  eval_tuned(fit_tune_kknn, "K-Nearest Neighbors", 18.7),
  eval_tuned(fit_tune_nb, "Naive Bayes", 5.2)
)
tuned_eval_df <- do.call(rbind, tuned_records)

tuning_comp <- baseline_df %>%
  filter(Model_ID %in% tuned_eval_df$Model_ID) %>%
  dplyr::select(Model_ID, Baseline_CV_Acc = CV_Accuracy, Baseline_Test_Acc = Accuracy, Baseline_F1 = Macro_F1) %>%
  inner_join(
    tuned_eval_df %>% dplyr::select(Model_ID, Best_Params, Tuned_CV_Acc = CV_Accuracy, Tuned_Test_Acc = Accuracy, Tuned_F1 = Macro_F1, Tuning_Time_Sec = Training_Time_Sec),
    by = "Model_ID"
  ) %>%
  mutate(
    CV_Improvement_Pct = paste0(round(((Tuned_CV_Acc - Baseline_CV_Acc) / Baseline_CV_Acc) * 100, 2), "%"),
    Test_Improvement_Pct = paste0(round(((Tuned_Test_Acc - Baseline_Test_Acc) / Baseline_Test_Acc) * 100, 2), "%")
  ) %>%
  arrange(desc(Tuned_Test_Acc))

print(tuning_comp)
write.csv(tuning_comp, file.path(tab_dir, "tuning_comparison.csv"), row.names = FALSE)
cat("--> Saved results/tables/tuning_comparison.csv\n")
cat("=== HYPERPARAMETER TUNING COMPLETE ===\n\n")
