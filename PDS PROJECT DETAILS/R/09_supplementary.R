# ==============================================================================
# 09_supplementary.R - Supplementary Experiments & Validation Benchmarks
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
cat("=== PART 3: SUPPLEMENTARY EXPERIMENTS & RIGOROUS BENCHMARKS ===\n")
cat("======================================================================\n\n")

tab_dir <- file.path(base_dir, "results/tables")
fig_dir <- file.path(base_dir, "results/figures")

train_df <- readRDS(file.path(tab_dir, "train_data.rds"))
test_df  <- readRDS(file.path(tab_dir, "test_data.rds"))
train_clean <- train_df[, !(colnames(train_df) %in% c("automation_risk_score"))]
test_clean  <- test_df[, !(colnames(test_df) %in% c("automation_risk_score"))]
y_test <- test_clean$risk_category

# Load supplementary tables created during analysis
supp_sel <- read.csv(file.path(tab_dir, "supplementary_selected_vs_all.csv"))
supp_abl <- read.csv(file.path(tab_dir, "supplementary_ablation.csv"))
supp_reg <- read.csv(file.path(tab_dir, "supplementary_regression.csv"))

cat("=== SUPPLEMENTARY EXPERIMENT (i): SELECTED FEATURES VS ALL VARIABLES ===\n")
print(supp_sel)

cat("\n=== SUPPLEMENTARY EXPERIMENT (ii): DOMINANCE CHECK & ABLATION STUDY ===\n")
print(supp_abl)

cat("\n=== SUPPLEMENTARY REGRESSION CHECK: CONTINUOUS TARGET ESTIMATION ===\n")
print(supp_reg)

cat("\n=== SUPPLEMENTARY EXPERIMENTS COMPLETE ===\n\n")
