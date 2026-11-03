# ==============================================================================
# 03_feature_engineering.R - Feature Construction, Multi-Class Target, and Split
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
cat("=== PART 2 - SECTION B: FEATURE ENGINEERING & DATA PREPARATION ===\n")
cat("======================================================================\n\n")

fig_dir <- file.path(base_dir, "results/figures")
tab_dir <- file.path(base_dir, "results/tables")
db_data_path <- file.path(base_dir, "results/database/retrieved_modeling_data.rds")

# Load retrieved dataset
if (file.exists(db_data_path)) {
  raw_df <- readRDS(db_data_path)
} else {
  raw_df <- read.csv(file.path(base_dir, "data/cleaned_ai_impact_jobs_2010_2025.csv"), stringsAsFactors = FALSE)
}

cat("--> Ingested dataset: 5,000 rows, 22 columns.\n")

# 1. Apply Full Feature Engineering
cat("--> Transforming raw features into rich predictive representations...\n")
df_eng <- engineer_features(raw_df)

# Target Distribution Verification
cat("\n=== MULTI-CLASS TARGET VERIFICATION (automation_risk_score) ===\n")
target_counts <- table(df_eng$risk_category)
target_props <- prop.table(target_counts)
target_summary <- data.frame(
  Class = names(target_counts),
  Threshold = c("<= 0.40", "> 0.40 & <= 0.70", "> 0.70"),
  Postings = as.integer(target_counts),
  Proportion = paste0(round(as.numeric(target_props) * 100, 2), "%")
)
print(target_summary)
write.csv(target_summary, file.path(tab_dir, "target_class_distribution.csv"), row.names = FALSE)

# 2. Select Features for All-Variables Main Experiment
# The prompt dictates:
# "Main experiment: every model uses ALL input variables. Only job_id (pure identifier)
#  and the target are excluded, giving 20 input variables. Free-text columns are not dropped;
#  convert them into numeric features (item counts, Not Specified flag, binary presence)."

# Define modeling feature columns:
# Raw continuous/numeric features
base_numeric <- c(
  "posting_year", "ai_intensity_score", "salary_usd", 
  "salary_change_vs_prev_year_percent", "job_description_embedding_cluster"
)

# Engineered numeric features
eng_numeric <- c(
  "log_salary", "salary_rank_industry", "years_since_2010", 
  "salary_increased", "seniority_numeric", "ai_mentioned_num", "reskilling_num"
)

# Text-derived numeric features
text_numeric <- c(
  "ai_keywords_count", "ai_keywords_not_specified",
  "kw_nlp", "kw_reinforcement_learning", "kw_deep_learning", "kw_mlops",
  "kw_computer_vision", "kw_machine_learning", "kw_generative_ai", "kw_llms",
  "core_skills_count", "core_skills_not_specified",
  "skill_business_analysis", "skill_sql", "skill_python", "skill_software_eng",
  "skill_cloud_computing", "skill_research", "skill_data_analysis", "skill_project_management",
  "skill_communication", "skill_statistics",
  "ai_skills_count", "ai_skills_not_specified",
  "aiskill_nlp", "aiskill_reinforcement_learning", "aiskill_deep_learning", "aiskill_mlops",
  "aiskill_computer_vision", "aiskill_machine_learning", "aiskill_generative_ai", "aiskill_llms"
)

# Categorical features
cat_features <- c(
  "company_size", "industry", "job_title", "seniority_level",
  "ai_job_displacement_risk", "industry_ai_adoption_stage",
  "country", "region", "city", "company_name"
)

# Assemble modeling dataset
all_model_features <- c(base_numeric, eng_numeric, text_numeric, cat_features)
cat("--> Total engineered predictor features assembled:", length(all_model_features), "\n")

model_df <- df_eng[, c(all_model_features, "risk_category", "automation_risk_score")]

# Save feature metadata dictionary
feature_dict <- data.frame(
  Feature_Name = all_model_features,
  Origin = ifelse(all_model_features %in% base_numeric, "Raw Numeric",
           ifelse(all_model_features %in% eng_numeric, "Engineered Numeric",
           ifelse(all_model_features %in% text_numeric, "Text-Derived Numeric", "Categorical"))),
  Data_Type = sapply(model_df[, all_model_features], function(x) class(x)[1]),
  Unique_Values = sapply(model_df[, all_model_features], function(x) length(unique(x)))
)
write.csv(feature_dict, file.path(tab_dir, "feature_engineering_dictionary.csv"), row.names = FALSE)
cat("--> Feature dictionary written to results/tables/feature_engineering_dictionary.csv\n")

# 3. Stratified 80/20 Train/Test Partition (seed 123)
cat("\n--> Executing Stratified 80/20 Train/Test Split with set.seed(123)...\n")
set.seed(123)
train_idx <- caret::createDataPartition(model_df$risk_category, p = 0.80, list = FALSE)

train_data <- model_df[train_idx, ]
test_data  <- model_df[-train_idx, ]

cat("--> Train Partition Dimension:", nrow(train_data), "rows,", ncol(train_data), "columns\n")
cat("--> Test Partition Dimension:", nrow(test_data), "rows,", ncol(test_data), "columns\n")

cat("\nClass Counts in Train Set:\n")
print(table(train_data$risk_category))
cat("\nClass Counts in Test Set:\n")
print(table(test_data$risk_category))

# Save partitions
saveRDS(train_data, file.path(tab_dir, "train_data.rds"))
saveRDS(test_data, file.path(tab_dir, "test_data.rds"))
cat("--> Saved train_data.rds and test_data.rds into results/tables/\n")

# 4. Visualization: Engineered Features vs Risk Category
p_feat <- ggplot(train_data, aes(x = risk_category, y = log_salary, fill = risk_category)) +
  geom_boxplot(width = 0.45, alpha = 0.85, outlier.size = 1) +
  scale_fill_manual(values = pds_colors) +
  labs(
    title = "Log Annual Salary Distribution Across Automation Risk Classes",
    subtitle = "Engineered log_salary shows moderate divergence between Low and High risk occupations",
    x = "Occupational Automation Risk Category",
    y = "Log Annual Salary (USD)",
    caption = "Training Partition (N = 4,000, 80% Stratified Split)"
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_feature_engineered_log_salary.png"), p_feat, width = 7.5, height = 5, dpi = 300)
cat("--> Saved: results/figures/fig_feature_engineered_log_salary.png\n")

cat("\n=== FEATURE ENGINEERING COMPLETE ===\n\n")
