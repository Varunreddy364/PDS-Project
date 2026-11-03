# ==============================================================================
# 08_visualizations.R - Comparative Visualizations (300 DPI Publication Quality)
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
cat("=== PART 3 - SECTION F: ADVANCED COMPARATIVE VISUALIZATIONS ===\n")
cat("======================================================================\n\n")

tab_dir <- file.path(base_dir, "results/tables")
fig_dir <- file.path(base_dir, "results/figures")
mod_dir <- file.path(base_dir, "results/models")

test_df  <- readRDS(file.path(tab_dir, "test_data.rds"))
test_clean  <- test_df[, !(colnames(test_df) %in% c("automation_risk_score"))]
y_test <- test_clean$risk_category

fit_tune_ranger <- readRDS(file.path(mod_dir, "tuned_random_forest.rds"))
fit_tune_gbm    <- readRDS(file.path(mod_dir, "tuned_gradient_boosting.rds"))
fit_tune_glmnet <- readRDS(file.path(mod_dir, "tuned_elastic_net.rds"))
fit_tune_rpart  <- readRDS(file.path(mod_dir, "tuned_decision_tree.rds"))
fit_tune_kknn   <- readRDS(file.path(mod_dir, "tuned_k_nearest_neighbors.rds"))

final_models_table <- read.csv(file.path(tab_dir, "final_model_comparison.csv"))
tuning_comp <- read.csv(file.path(tab_dir, "tuning_comparison.csv"))

# 1. Metric Heatmap Across All Models
cat("--> Generating Metric Heatmap for All Models...\n")
metric_cols <- c("Accuracy", "Balanced_Accuracy", "Kappa", "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_AUC")
heatmap_df <- final_models_table %>%
  filter(Model_ID != "Majority Class Benchmark") %>%
  dplyr::select(Model_ID, all_of(metric_cols)) %>%
  pivot_longer(cols = all_of(metric_cols), names_to = "Metric", values_to = "Score")

p_heat <- ggplot(heatmap_df, aes(x = Metric, y = reorder(Model_ID, Score), fill = Score)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Score)), fontface = "bold", size = 3.2, color = "#1F2D3D") +
  scale_fill_gradient(low = "#EBF0F5", high = "#008991") +
  labs(
    title = "Comprehensive Metric Heatmap Across All Evaluated Models",
    subtitle = "Multi-class performance metrics evaluated on held-out test dataset (N = 998)",
    x = "Performance Metric", y = "Model Architecture"
  ) +
  theme_pds() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(file.path(fig_dir, "fig_metric_heatmap.png"), p_heat, width = 10, height = 6.5, dpi = 300)
cat("--> Saved: results/figures/fig_metric_heatmap.png\n")

# 2. Resamples Cross-Validation Boxplots
cat("--> Generating CV Boxplots Across Model Resamples...\n")
models_list <- list(
  "Random Forest" = tryCatch(readRDS(file.path(mod_dir, "tuned_random_forest.rds")), error = function(e) NULL),
  "Elastic-Net"   = tryCatch(readRDS(file.path(mod_dir, "tuned_elastic_net.rds")), error = function(e) NULL),
  "Decision Tree" = tryCatch(readRDS(file.path(mod_dir, "tuned_decision_tree.rds")), error = function(e) NULL),
  "GBM"           = tryCatch(readRDS(file.path(mod_dir, "tuned_gradient_boosting.rds")), error = function(e) NULL),
  "Naive Bayes"   = tryCatch(readRDS(file.path(mod_dir, "tuned_naive_bayes.rds")), error = function(e) NULL)
)
models_list <- models_list[!sapply(models_list, is.null)]
resamps <- caret::resamples(models_list)
resamp_df <- resamps$values %>%
  pivot_longer(cols = -Resample, names_to = "Model_Metric", values_to = "Value") %>%
  separate(Model_Metric, into = c("Model", "Metric"), sep = "~")

p_box <- ggplot(resamp_df %>% filter(Metric %in% c("Accuracy", "Kappa")), aes(x = reorder(Model, Value, FUN = median), y = Value, fill = Model)) +
  geom_boxplot(alpha = 0.8, outlier.size = 1.2, show.legend = FALSE) +
  facet_wrap(~ Metric, scales = "free_y") +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Cross-Validation Stability: Repeated 5-Fold Resample Distributions",
    subtitle = "Accuracy and Kappa variance across 15 resamples (5-fold CV, 3 repeats)",
    x = "Model Architecture", y = "Resample Metric Value"
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_cv_boxplots.png"), p_box, width = 9.5, height = 5.5, dpi = 300)
cat("--> Saved: results/figures/fig_cv_boxplots.png\n")

cat("=== COMPARATIVE VISUALIZATIONS COMPLETE ===\n\n")
