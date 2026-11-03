# ==============================================================================
# 07_comparison.R - Comprehensive Comparative Performance & Statistical Resamples
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
cat("=== PART 3 - SECTION E: COMPARATIVE PERFORMANCE & STATISTICAL TESTS ===\n")
cat("======================================================================\n\n")

tab_dir <- file.path(base_dir, "results/tables")
mod_dir <- file.path(base_dir, "results/models")

baseline_df <- read.csv(file.path(tab_dir, "baseline_models_performance.csv"))
tuning_comp <- read.csv(file.path(tab_dir, "tuning_comparison.csv"))

# Build master ranking table
final_models_table <- read.csv(file.path(tab_dir, "final_model_comparison.csv"))
print(final_models_table[, c("Model_ID", "Accuracy", "Balanced_Accuracy", "Kappa", "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_AUC", "Log_Loss", "Training_Time_Sec")])

# Statistical Significance Comparison via caret::resamples
cat("\n--> Conducting Statistical Resamples Comparison (resamples() & diff())...\n")

# Load fitted caret objects for comparison (tuned models)
fit_ranger <- tryCatch(readRDS(file.path(mod_dir, "tuned_random_forest.rds")), error = function(e) NULL)
fit_glmnet <- tryCatch(readRDS(file.path(mod_dir, "tuned_elastic_net.rds")), error = function(e) NULL)
fit_rpart  <- tryCatch(readRDS(file.path(mod_dir, "tuned_decision_tree.rds")), error = function(e) NULL)
fit_gbm    <- tryCatch(readRDS(file.path(mod_dir, "tuned_gradient_boosting.rds")), error = function(e) NULL)
fit_nb     <- tryCatch(readRDS(file.path(mod_dir, "tuned_naive_bayes.rds")), error = function(e) NULL)

models_list <- list(
  Random_Forest = fit_ranger,
  Elastic_Net   = fit_glmnet,
  Decision_Tree = fit_rpart,
  GBM           = fit_gbm,
  Naive_Bayes   = fit_nb
)
models_list <- models_list[!sapply(models_list, is.null)]

if (length(models_list) >= 2) {
  resamps <- caret::resamples(models_list)
  cat("Resamples Summary (Accuracy & Kappa):\n")
  print(summary(resamps, metric = c("Accuracy", "Kappa")))

  # Compute paired differences specifically on Accuracy
  diffs <- diff(resamps, metric = "Accuracy")
  cat("\nPaired Differences Matrix on Accuracy (p-values with Bonferroni correction):\n")
  print(summary(diffs))

  sink(file.path(tab_dir, "statistical_significance_summary.txt"))
  cat("=== CARET RESAMPLES STATISTICAL SIGNIFICANCE SUMMARY ===\n\n")
  print(summary(resamps, metric = c("Accuracy", "Kappa")))
  cat("\n=== PAIRED DIFFERENCES ON ACCURACY (diff.resamples) ===\n\n")
  print(summary(diffs))
  sink()
  cat("--> Statistical significance summary saved to results/tables/statistical_significance_summary.txt\n")
}
cat("=== COMPARATIVE PERFORMANCE ANALYSIS COMPLETE ===\n\n")
