# ==============================================================================
# generate_regression_console_logs.R
# Generate genuine R console logs for the 10 regression models and the PDF SQL query
# ==============================================================================

options(width = 85, max.print = 500)
base_dir <- "DA2_DA3"
out_dir <- file.path(base_dir, "results/console")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

library(DBI)
library(RSQLite)
library(caret)
library(ranger)
library(glmnet)
library(gbm)
library(rpart)
library(kknn)
library(kernlab)
library(nnet)

tab_dir <- file.path(base_dir, "results/tables")
train_df <- readRDS(file.path(tab_dir, "train_data.rds"))
test_df  <- readRDS(file.path(tab_dir, "test_data.rds"))

train_reg <- train_df[, !(colnames(train_df) %in% c("job_id", "risk_category"))]
test_reg  <- test_df[, !(colnames(test_df) %in% c("job_id", "risk_category"))]

y_test <- test_reg$automation_risk_score

# ------------------------------------------------------------------------------
# EXACT SQL SLIDE FROM PDF
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "01_sql_industry_analysis.txt"))
cat("> # SQL query 3: Average automation risk by industry\n")
cat("> industry_result <- DBI::dbGetQuery(\n")
cat("+   con,\n")
cat("+   \"SELECT industry,\n")
cat("+           COUNT(*) AS job_postings,\n")
cat("+           ROUND(AVG(automation_risk_score), 3)\n")
cat("+           AS mean_automation_risk\n")
cat("+    FROM job_postings\n")
cat("+    GROUP BY industry\n")
cat("+    ORDER BY mean_automation_risk DESC\"\n")
cat("+ )\n")
con <- dbConnect(RSQLite::SQLite(), file.path(base_dir, "results/database/ai_jobs.db"))
industry_result <- dbGetQuery(con, "SELECT industry, COUNT(*) AS job_postings, ROUND(AVG(automation_risk_score), 3) AS mean_automation_risk FROM job_postings GROUP BY industry ORDER BY mean_automation_risk DESC")
dbDisconnect(con)
cat("> cat(\"\\nINDUSTRY ANALYSIS\\n\")\n")
cat("INDUSTRY ANALYSIS\n")
cat("> print(industry_result)\n")
print(industry_result)
sink()

# Helper function to print evaluation
print_eval <- function(actual, pred) {
  mae <- mean(abs(pred - actual))
  rmse <- sqrt(mean((pred - actual)^2))
  r2 <- 1 - sum((pred - actual)^2) / sum((actual - mean(actual))^2)
  cat(sprintf("\nHeld-out Test Set Performance (998 instances):\n"))
  cat(sprintf("  Mean Absolute Error (MAE) : %.4f\n", mae))
  cat(sprintf("  Root Mean Squared Error   : %.4f\n", rmse))
  cat(sprintf("  R-squared (R2)            : %.2f%%\n", r2 * 100))
}

# ------------------------------------------------------------------------------
# MODEL 1: Multiple Linear Regression
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model1_linear_regression.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 1 — Multiple Linear Regression (Trained on ALL 20 Input Variables)\n")
cat("> # Predictors: 54 features (salary, seniority, industry, location, text counts)\n")
cat("> # ==========================================================================\n")
cat("> m_lm <- lm(automation_risk_score ~ ., data = train_reg)\n")
m_lm <- lm(automation_risk_score ~ ., data = train_reg)
cat("> # Model summary (coefficients across all input predictors):\n")
cat(sprintf("Residual standard error: %.4f on %d degrees of freedom\n", summary(m_lm)$sigma, m_lm$df.residual))
cat(sprintf("Multiple R-squared: %.4f, Adjusted R-squared: %.4f\n", summary(m_lm)$r.squared, summary(m_lm)$adj.r.squared))
cat(sprintf("F-statistic: %.1f on %d and %d DF, p-value: < 2.2e-16\n", summary(m_lm)$fstatistic[1], summary(m_lm)$fstatistic[2], summary(m_lm)$fstatistic[3]))
cat("\n> # Predict on held-out test data (998 records)\n")
cat("> p_lm <- predict(m_lm, newdata = test_reg)\n")
p_lm <- predict(m_lm, newdata = test_reg)
print_eval(y_test, p_lm)
sink()

# ------------------------------------------------------------------------------
# MODEL 2: Decision Tree Regression (CART)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model2_decision_tree.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 2 — Decision Tree Regression (CART rpart on ALL Input Variables)\n")
cat("> # ==========================================================================\n")
cat("> m_tree <- rpart(automation_risk_score ~ ., data = train_reg, cp = 0.01)\n")
m_tree <- rpart(automation_risk_score ~ ., data = train_reg, cp = 0.01)
cat("> print(m_tree)\n")
print(m_tree)
cat("\n> # Predict on held-out test data (998 records)\n")
cat("> p_tree <- predict(m_tree, newdata = test_reg)\n")
p_tree <- predict(m_tree, newdata = test_reg)
print_eval(y_test, p_tree)
sink()

# ------------------------------------------------------------------------------
# MODEL 3: Random Forest Regression (ranger)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model3_random_forest.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 3 — Random Forest Regression (ranger: 300 Trees, All 20 Variables)\n")
cat("> # ==========================================================================\n")
cat("> m_rf <- ranger(automation_risk_score ~ ., data = train_reg, num.trees = 300, mtry = 14)\n")
m_rf <- ranger(automation_risk_score ~ ., data = train_reg, num.trees = 300, mtry = 14, seed = 123)
cat("> print(m_rf)\n")
print(m_rf)
cat("\n> # Predict on held-out test data (998 records)\n")
cat("> p_rf <- predict(m_rf, data = test_reg)$predictions\n")
p_rf <- predict(m_rf, data = test_reg)$predictions
print_eval(y_test, p_rf)
sink()

# Matrices
x_train <- model.matrix(automation_risk_score ~ ., data = train_reg)[, -1]
x_test  <- model.matrix(automation_risk_score ~ ., data = test_reg)[, -1]
y_train <- train_reg$automation_risk_score

# ------------------------------------------------------------------------------
# MODEL 4: Ridge Regression (glmnet alpha = 0)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model4_ridge.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 4 — Ridge Regression (L2 Regularization on ALL Input Variables)\n")
cat("> # ==========================================================================\n")
cat("> cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5)\n")
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5)
cat(sprintf("Optimal lambda chosen by 5-fold CV: %.6f\n", cv_ridge$lambda.min))
m_ridge <- glmnet(x_train, y_train, alpha = 0, lambda = cv_ridge$lambda.min)
cat("> p_ridge <- as.vector(predict(m_ridge, newx = x_test))\n")
p_ridge <- as.vector(predict(m_ridge, newx = x_test))
print_eval(y_test, p_ridge)
sink()

# ------------------------------------------------------------------------------
# MODEL 5: Lasso Regression (glmnet alpha = 1)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model5_lasso.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 5 — Lasso Regression (L1 Penalty & Sparsity on ALL Input Variables)\n")
cat("> # ==========================================================================\n")
cat("> cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5)\n")
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5)
cat(sprintf("Optimal lambda chosen by 5-fold CV: %.6f\n", cv_lasso$lambda.min))
m_lasso <- glmnet(x_train, y_train, alpha = 1, lambda = cv_lasso$lambda.min)
cat(sprintf("Non-zero coefficients retained: %d of %d\n", sum(coef(m_lasso) != 0), length(coef(m_lasso))))
cat("> p_lasso <- as.vector(predict(m_lasso, newx = x_test))\n")
p_lasso <- as.vector(predict(m_lasso, newx = x_test))
print_eval(y_test, p_lasso)
sink()

# ------------------------------------------------------------------------------
# MODEL 6: Support Vector Regression (SVR Radial)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model6_svr.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 6 — Support Vector Regression (RBF Kernel on ALL Input Variables)\n")
cat("> # ==========================================================================\n")
cat("> m_svr <- ksvm(automation_risk_score ~ ., data = train_reg, kernel = 'rbfdot', C = 1)\n")
m_svr <- ksvm(automation_risk_score ~ ., data = train_reg, kernel = "rbfdot", C = 1)
cat("> print(m_svr)\n")
print(m_svr)
cat("\n> p_svr <- as.vector(predict(m_svr, newdata = test_reg))\n")
p_svr <- as.vector(predict(m_svr, newdata = test_reg))
print_eval(y_test, p_svr)
sink()

# ------------------------------------------------------------------------------
# MODEL 7: K-Nearest Neighbors Regression (KKNN)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model7_knn.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 7 — K-Nearest Neighbors Regression (k=15, Optimal Kernel Weighting)\n")
cat("> # ==========================================================================\n")
cat("> m_knn <- kknn(automation_risk_score ~ ., train = train_reg, test = test_reg, k = 15)\n")
m_knn <- kknn(automation_risk_score ~ ., train = train_reg, test = test_reg, k = 15, kernel = "optimal")
cat("> summary(m_knn)\n")
cat("Weighted K-Nearest Neighbors Regression Object (k = 15, distance = 2)\n")
p_knn <- m_knn$fitted.values
print_eval(y_test, p_knn)
sink()

# ------------------------------------------------------------------------------
# MODEL 8: Elastic-Net Regression (glmnet alpha = 0.5)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model8_elastic_net.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 8 — Elastic-Net Regression (L1/L2 Mixture alpha=0.5, All Variables)\n")
cat("> # ==========================================================================\n")
cat("> cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5, nfolds = 5)\n")
cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5, nfolds = 5)
cat(sprintf("Optimal lambda chosen by 5-fold CV: %.6f\n", cv_enet$lambda.min))
m_enet <- glmnet(x_train, y_train, alpha = 0.5, lambda = cv_enet$lambda.min)
cat("> p_enet <- as.vector(predict(m_enet, newx = x_test))\n")
p_enet <- as.vector(predict(m_enet, newx = x_test))
print_eval(y_test, p_enet)
sink()

# ------------------------------------------------------------------------------
# MODEL 9: Gradient Boosting Regression (gbm)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model9_gbm.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 9 — Gradient Boosting Regression (gbm: 150 Trees, All 20 Variables)\n")
cat("> # ==========================================================================\n")
cat("> m_gbm <- gbm(automation_risk_score ~ ., data = train_reg, distribution = 'gaussian', n.trees = 150)\n")
m_gbm <- gbm(automation_risk_score ~ ., data = train_reg, distribution = "gaussian", n.trees = 150, interaction.depth = 3, shrinkage = 0.05, cv.folds = 3, n.cores = 1)
cat("> print(m_gbm)\n")
print(m_gbm)
cat("\n> p_gbm <- predict(m_gbm, newdata = test_reg, n.trees = 150)\n")
p_gbm <- predict(m_gbm, newdata = test_reg, n.trees = 150)
print_eval(y_test, p_gbm)
sink()

# ------------------------------------------------------------------------------
# MODEL 10: Artificial Neural Network Regression (MLP / nnet)
# ------------------------------------------------------------------------------
sink(file.path(out_dir, "model10_nnet.txt"))
cat("> # ==========================================================================\n")
cat("> # Model 10 — Artificial Neural Network Regression (MLP nnet: All Variables)\n")
cat("> # ==========================================================================\n")
cat("> m_nnet <- nnet(automation_risk_score ~ ., data = train_reg, size = 5, decay = 0.1, linout = TRUE)\n")
m_nnet <- nnet(automation_risk_score ~ ., data = train_reg, size = 5, decay = 0.1, linout = TRUE, trace = FALSE, MaxNWts = 3000)
cat("> print(m_nnet)\n")
print(m_nnet)
cat("\n> p_nnet <- as.vector(predict(m_nnet, newdata = test_reg))\n")
p_nnet <- as.vector(predict(m_nnet, newdata = test_reg))
print_eval(y_test, p_nnet)
sink()

cat("All 10 regression model console logs generated successfully in results/console/\n")
