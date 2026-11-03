

library(caret)
library(ranger)
library(glmnet)
library(gbm)
library(rpart)
library(kknn)
library(kernlab)
library(nnet)

base_dir <- "DA2_DA3"
tab_dir <- file.path(base_dir, "results/tables")
mod_dir <- file.path(base_dir, "results/models")
con_dir <- file.path(base_dir, "results/console")

train_df <- readRDS(file.path(tab_dir, "train_data.rds"))
test_df  <- readRDS(file.path(tab_dir, "test_data.rds"))

# All 20 input variables: exclude job_id and risk_category, keep automation_risk_score as target
train_reg <- train_df[, !(colnames(train_df) %in% c("job_id", "risk_category"))]
test_reg  <- test_df[, !(colnames(test_df) %in% c("job_id", "risk_category"))]

y_train <- train_reg$automation_risk_score
y_test  <- test_reg$automation_risk_score

cat("--> Training regression models on all", ncol(train_reg) - 1, "predictor features...\n")
cat("--> Train set:", nrow(train_reg), "rows | Test set:", nrow(test_reg), "rows\n\n")

eval_reg <- function(actual, predicted, train_time, model_name) {
  mae <- mean(abs(predicted - actual))
  rmse <- sqrt(mean((predicted - actual)^2))
  r2 <- 1 - sum((predicted - actual)^2) / sum((actual - mean(actual))^2)
  cat(sprintf("[%s] MAE: %.4f | RMSE: %.4f | R2: %.2f%% | Time: %.2fs\n",
              model_name, mae, rmse, r2 * 100, train_time))
  data.frame(
    Model = model_name,
    MAE = round(mae, 4),
    RMSE = round(rmse, 4),
    R2_Pct = round(r2 * 100, 2),
    Time_Sec = round(train_time, 2)
  )
}

results_list <- list()

# ------------------------------------------------------------------------------
# 1. Multiple Linear Regression
# ------------------------------------------------------------------------------
cat("--> Training 1. Multiple Linear Regression...\n")
t0 <- Sys.time()
m_lm <- lm(automation_risk_score ~ ., data = train_reg)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_lm <- predict(m_lm, newdata = test_reg)
results_list[["LM"]] <- eval_reg(y_test, p_lm, t1, "Multiple Linear Regression")
saveRDS(m_lm, file.path(mod_dir, "reg_model1_multiple_linear.rds"))

# ------------------------------------------------------------------------------
# 2. Decision Tree Regression (CART)
# ------------------------------------------------------------------------------
cat("--> Training 2. Decision Tree Regression...\n")
t0 <- Sys.time()
m_tree <- rpart(automation_risk_score ~ ., data = train_reg, cp = 0.01)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_tree <- predict(m_tree, newdata = test_reg)
results_list[["Tree"]] <- eval_reg(y_test, p_tree, t1, "Decision Tree Regression")
saveRDS(m_tree, file.path(mod_dir, "reg_model2_decision_tree.rds"))

# ------------------------------------------------------------------------------
# 3. Random Forest Regression (ranger)
# ------------------------------------------------------------------------------
cat("--> Training 3. Random Forest Regression...\n")
t0 <- Sys.time()
m_rf <- ranger(automation_risk_score ~ ., data = train_reg, num.trees = 300, mtry = 14, min.node.size = 5, seed = 123)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_rf <- predict(m_rf, data = test_reg)$predictions
results_list[["RF"]] <- eval_reg(y_test, p_rf, t1, "Random Forest Regression")
saveRDS(m_rf, file.path(mod_dir, "reg_model3_random_forest.rds"))

# Prepare matrix for glmnet and distance models
x_train <- model.matrix(automation_risk_score ~ ., data = train_reg)[, -1]
x_test  <- model.matrix(automation_risk_score ~ ., data = test_reg)[, -1]

# ------------------------------------------------------------------------------
# 4. Ridge Regression (glmnet alpha = 0)
# ------------------------------------------------------------------------------
cat("--> Training 4. Ridge Regression...\n")
t0 <- Sys.time()
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5)
m_ridge <- glmnet(x_train, y_train, alpha = 0, lambda = cv_ridge$lambda.min)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_ridge <- as.vector(predict(m_ridge, newx = x_test))
results_list[["Ridge"]] <- eval_reg(y_test, p_ridge, t1, "Ridge Regression")
saveRDS(m_ridge, file.path(mod_dir, "reg_model4_ridge.rds"))

# ------------------------------------------------------------------------------
# 5. Lasso Regression (glmnet alpha = 1)
# ------------------------------------------------------------------------------
cat("--> Training 5. Lasso Regression...\n")
t0 <- Sys.time()
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5)
m_lasso <- glmnet(x_train, y_train, alpha = 1, lambda = cv_lasso$lambda.min)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_lasso <- as.vector(predict(m_lasso, newx = x_test))
results_list[["Lasso"]] <- eval_reg(y_test, p_lasso, t1, "Lasso Regression")
saveRDS(m_lasso, file.path(mod_dir, "reg_model5_lasso.rds"))

# ------------------------------------------------------------------------------
# 6. Support Vector Regression (SVR Radial)
# ------------------------------------------------------------------------------
cat("--> Training 6. Support Vector Regression...\n")
t0 <- Sys.time()
m_svr <- ksvm(automation_risk_score ~ ., data = train_reg, kernel = "rbfdot", C = 1, kpar = "automatic")
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_svr <- as.vector(predict(m_svr, newdata = test_reg))
results_list[["SVR"]] <- eval_reg(y_test, p_svr, t1, "Support Vector Regression")
saveRDS(m_svr, file.path(mod_dir, "reg_model6_svr.rds"))

# ------------------------------------------------------------------------------
# 7. K-Nearest Neighbors Regression (KKNN)
# ------------------------------------------------------------------------------
cat("--> Training 7. K-Nearest Neighbors Regression...\n")
t0 <- Sys.time()
m_knn <- kknn(automation_risk_score ~ ., train = train_reg, test = test_reg, k = 15, kernel = "optimal")
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_knn <- m_knn$fitted.values
results_list[["KNN"]] <- eval_reg(y_test, p_knn, t1, "K-Nearest Neighbors Regression")
saveRDS(m_knn, file.path(mod_dir, "reg_model7_knn.rds"))

# ------------------------------------------------------------------------------
# 8. Elastic-Net Regression (glmnet alpha = 0.5)
# ------------------------------------------------------------------------------
cat("--> Training 8. Elastic-Net Regression...\n")
t0 <- Sys.time()
cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5, nfolds = 5)
m_enet <- glmnet(x_train, y_train, alpha = 0.5, lambda = cv_enet$lambda.min)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_enet <- as.vector(predict(m_enet, newx = x_test))
results_list[["ElasticNet"]] <- eval_reg(y_test, p_enet, t1, "Elastic-Net Regression")
saveRDS(m_enet, file.path(mod_dir, "reg_model8_elastic_net.rds"))

# ------------------------------------------------------------------------------
# 9. Gradient Boosting Regression (gbm)
# ------------------------------------------------------------------------------
cat("--> Training 9. Gradient Boosting Regression...\n")
t0 <- Sys.time()
m_gbm <- gbm(automation_risk_score ~ ., data = train_reg, distribution = "gaussian", n.trees = 150, interaction.depth = 3, shrinkage = 0.05, cv.folds = 3, n.cores = 1)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_gbm <- predict(m_gbm, newdata = test_reg, n.trees = 150)
results_list[["GBM"]] <- eval_reg(y_test, p_gbm, t1, "Gradient Boosting Regression")
saveRDS(m_gbm, file.path(mod_dir, "reg_model9_gbm.rds"))

# ------------------------------------------------------------------------------
# 10. Artificial Neural Network Regression (MLP / nnet)
# ------------------------------------------------------------------------------
cat("--> Training 10. Neural Network Regression...\n")
t0 <- Sys.time()
m_nnet <- nnet(automation_risk_score ~ ., data = train_reg, size = 5, decay = 0.1, linout = TRUE, trace = FALSE, MaxNWts = 3000)
t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p_nnet <- as.vector(predict(m_nnet, newdata = test_reg))
results_list[["NNET"]] <- eval_reg(y_test, p_nnet, t1, "Neural Network Regression")
saveRDS(m_nnet, file.path(mod_dir, "reg_model10_nnet.rds"))

# Master summary table
df_master <- do.call(rbind, results_list)
df_master$Rank <- rank(df_master$MAE)
df_master <- df_master[order(df_master$MAE), ]
print(df_master)
write.csv(df_master, file.path(tab_dir, "ten_regression_models_performance.csv"), row.names = FALSE)
cat("\n--> Successfully saved ten_regression_models_performance.csv\n")
