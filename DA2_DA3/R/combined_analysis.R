# ==============================================================================
# combined_analysis.R - Consolidated End-to-End Analysis for Parts 2 & 3
# Authors: Karepalli Varun Reddy (24BDS1126) & Madheshwaran P (24BDS1159)
# Course: Programming for Data Science, VIT Chennai
# ==============================================================================

if (file.exists("DA2_DA3/R/functions.R")) {
  base_dir <- "DA2_DA3"
} else {
  base_dir <- "."
}

source(file.path(base_dir, "R/functions.R"))
ensure_packages()

cat("Executing Consolidated End-to-End Analysis for DA2 & DA3...\n")
source(file.path(base_dir, "R/01_database.R"))
source(file.path(base_dir, "R/02_data_audit.R"))
source(file.path(base_dir, "R/03_feature_engineering.R"))
source(file.path(base_dir, "R/04_feature_selection.R"))
source(file.path(base_dir, "R/06_tuning.R"))
source(file.path(base_dir, "R/07_comparison.R"))
source(file.path(base_dir, "R/08_visualizations.R"))
source(file.path(base_dir, "R/09_supplementary.R"))
source(file.path(base_dir, "R/10_progress.R"))
cat("\nConsolidated analysis complete.\n")
