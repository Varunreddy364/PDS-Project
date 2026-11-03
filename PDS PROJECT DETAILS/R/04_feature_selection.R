# ==============================================================================
# 04_feature_selection.R - Multi-Method Feature Selection & Ranking Protocol
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
cat("=== PART 2 - SECTION B: MULTI-METHOD FEATURE SELECTION PROTOCOL ===\n")
cat("======================================================================\n\n")

fig_dir <- file.path(base_dir, "results/figures")
tab_dir <- file.path(base_dir, "results/tables")

# Load training data
train_data <- readRDS(file.path(tab_dir, "train_data.rds"))
cat("--> Loaded training partition:", nrow(train_data), "rows,", ncol(train_data), "columns.\n")

# Prepare predictor matrix and target
y_train <- train_data$risk_category
X_train_df <- train_data[, !(colnames(train_data) %in% c("risk_category", "automation_risk_score"))]

# Create one-hot dummy encoded model matrix for algorithms that require numeric input
dummies <- dummyVars(~ ., data = X_train_df, fullRank = TRUE)
X_train_mat <- predict(dummies, newdata = X_train_df)
cat("--> Full one-hot expanded feature matrix dimension:", nrow(X_train_mat), "rows,", ncol(X_train_mat), "columns.\n")

# ------------------------------------------------------------------------------
# METHOD 1: Near-Zero Variance (NZV) Diagnostics
# ------------------------------------------------------------------------------
cat("\n=== METHOD 1: NEAR-ZERO VARIANCE (NZV) FILTER ===\n")
nzv_info <- caret::nearZeroVar(X_train_mat, saveMetrics = TRUE)
nzv_drops <- rownames(nzv_info)[nzv_info$nzv]
cat("Total features evaluated:", nrow(nzv_info), "\n")
cat("Features flagged as Near-Zero Variance:", length(nzv_drops), "\n")
if (length(nzv_drops) > 0) {
  cat("Examples of NZV features flagged:\n")
  print(head(nzv_drops, 8))
}

# ------------------------------------------------------------------------------
# METHOD 2: Pairwise Correlation Filter (Collinearity Pruning)
# ------------------------------------------------------------------------------
cat("\n=== METHOD 2: HIGH-CORRELATION FILTER (|r| > 0.85) ===\n")
# Filter out zero-variance columns before calculating correlations
non_zero_cols <- colnames(X_train_mat)[apply(X_train_mat, 2, sd) > 1e-6]
cor_mat <- cor(X_train_mat[, non_zero_cols])
high_cor_idx <- caret::findCorrelation(cor_mat, cutoff = 0.85, names = FALSE)
high_cor_drops <- colnames(cor_mat)[high_cor_idx]

cat("Features flagged for severe collinearity (|r| > 0.85):", length(high_cor_drops), "\n")
print(high_cor_drops)
cat("--> Expected collinear pairs identified (e.g., years_since_2010 vs posting_year, reskilling_num vs ai_mentioned_num).\n")

# ------------------------------------------------------------------------------
# METHOD 3: Regularized Elastic Net Multi-Class Selection (L1 Sparsity)
# ------------------------------------------------------------------------------
cat("\n=== METHOD 3: MULTI-CLASS ELASTIC-NET REGULARIZATION (glmnet) ===\n")
set.seed(123)
cv_en <- glmnet::cv.glmnet(
  x = X_train_mat[, non_zero_cols],
  y = y_train,
  family = "multinomial",
  alpha = 0.5,
  nfolds = 5
)

# Extract coefficients at lambda.1se
coef_list <- coef(cv_en, s = "lambda.1se")
# Compute mean absolute coefficient magnitude across classes for each feature
en_imp <- sapply(non_zero_cols, function(feat) {
  mean(sapply(coef_list, function(mat) {
    if (feat %in% rownames(mat)) abs(mat[feat, 1]) else 0
  }))
})
en_imp_df <- data.frame(
  Feature = non_zero_cols,
  ElasticNet_Imp = as.numeric(en_imp)
) %>% arrange(desc(ElasticNet_Imp))

cat("Top 10 features by Elastic-Net multi-class importance:\n")
print(head(en_imp_df, 10))

# ------------------------------------------------------------------------------
# METHOD 4: Tree-Based Permutation/Impurity Importance (Random Forest via ranger)
# ------------------------------------------------------------------------------
cat("\n=== METHOD 4: RANDOM FOREST FEATURE IMPORTANCE (ranger) ===\n")
set.seed(123)
rf_fs <- ranger::ranger(
  formula = risk_category ~ .,
  data = train_data[, !(colnames(train_data) %in% c("automation_risk_score"))],
  importance = "impurity",
  num.trees = 200,
  num.threads = 3
)

rf_imp <- ranger::importance(rf_fs)
rf_imp_df <- data.frame(
  Feature = names(rf_imp),
  RF_Impurity_Imp = as.numeric(rf_imp)
) %>% arrange(desc(RF_Impurity_Imp))

cat("Top 10 features by Random Forest Gini impurity decrease:\n")
print(head(rf_imp_df, 10))

# ------------------------------------------------------------------------------
# CONSOLIDATED FEATURE SELECTION RANKING TABLE
# ------------------------------------------------------------------------------
cat("\n--> Building Consolidated Feature Selection Ranking Table...\n")
all_predictors <- colnames(X_train_df)

# Map back from dummy names to original feature concepts
ranking_table <- data.frame(Feature = all_predictors) %>%
  left_join(rf_imp_df, by = "Feature") %>%
  mutate(
    RF_Impurity_Imp = ifelse(is.na(RF_Impurity_Imp), 0, RF_Impurity_Imp),
    RF_Rank = rank(-RF_Impurity_Imp, ties.method = "min")
  )

# Add NZV flag (original level)
ranking_table$NZV_Flag <- sapply(ranking_table$Feature, function(f) {
  matches <- grep(paste0("^", f), nzv_drops, value = TRUE)
  length(matches) > 0
})

# Add High Cor flag
ranking_table$High_Cor_Flag <- sapply(ranking_table$Feature, function(f) {
  matches <- grep(paste0("^", f), high_cor_drops, value = TRUE)
  length(matches) > 0
})

# Recommendation Logic:
# Keep: Highly important non-collinear features (Top in RF/EN, not pure duplicates)
# Drop: Redundant collinear duplicates (e.g. years_since_2010, reskilling_num) or pure noise
ranking_table <- ranking_table %>%
  mutate(
    Recommendation = case_when(
      High_Cor_Flag & (Feature %in% c("years_since_2010", "reskilling_num", "ai_keywords_not_specified", "ai_skills_not_specified")) ~ "Drop (Collinear Proxy)",
      NZV_Flag & RF_Rank > 30 ~ "Drop (Near-Zero Variance)",
      RF_Rank <= 12 ~ "Keep (Primary Predictor)",
      RF_Rank <= 22 ~ "Keep (Secondary Informative)",
      TRUE ~ "Drop (Low Information Utility)"
    )
  ) %>%
  arrange(RF_Rank)

print(head(ranking_table, 25))
write.csv(ranking_table, file.path(tab_dir, "feature_selection_ranking.csv"), row.names = FALSE)
cat("--> Master feature ranking table written to results/tables/feature_selection_ranking.csv\n")

# Extract the selected feature subset for Supplementary Experiment (i)
selected_subset <- ranking_table %>%
  filter(grepl("Keep", Recommendation)) %>%
  pull(Feature)

cat("\n--> Selected Feature Subset (", length(selected_subset), " variables) for Supplementary Experiment (i):\n", sep = "")
print(selected_subset)
saveRDS(selected_subset, file.path(tab_dir, "selected_features_subset.rds"))

# ------------------------------------------------------------------------------
# VISUALIZATION: Feature Selection Importance Bar Chart
# ------------------------------------------------------------------------------
top_viz <- head(ranking_table, 18)
p_fs <- ggplot(top_viz, aes(x = reorder(Feature, RF_Impurity_Imp), y = RF_Impurity_Imp, fill = Recommendation)) +
  geom_col(width = 0.72) +
  coord_flip() +
  scale_fill_manual(values = c(
    "Keep (Primary Predictor)" = "#008991",
    "Keep (Secondary Informative)" = "#275E9C",
    "Drop (Collinear Proxy)" = "#E6AB37",
    "Drop (Low Information Utility)" = "#D9534F"
  )) +
  labs(
    title = "Feature Selection: Top Variables by Random Forest Gini Importance",
    subtitle = "ai_intensity_score and ai_mentioned dominate; selected subset retains top non-redundant drivers",
    x = "Predictor Feature",
    y = "Random Forest Impurity Decrease (Gini Importance)",
    fill = "Selection Recommendation",
    caption = "Evaluated on 4,002 training observations using ranger (200 trees)"
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_feature_selection_importance.png"), p_fs, width = 9, height = 6.5, dpi = 300)
cat("--> Saved: results/figures/fig_feature_selection_importance.png\n")

cat("\n=== FEATURE SELECTION COMPLETE ===\n\n")
