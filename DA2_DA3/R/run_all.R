# ==============================================================================
# run_all.R - Master Pipeline Driver for DA2 & DA3 Project (VIT Chennai)
# Authors: Karepalli Varun Reddy (24BDS1126) & Madheshwaran P (24BDS1159)
# Course: Programming for Data Science
# "Will AI Replace Your Job? A machine-learning analysis of occupational AI risk"
# ==============================================================================

cat("\n")
cat("################################################################################\n")
cat("###   WILL AI REPLACE YOUR JOB? OCCUPATIONAL AUTOMATION RISK ANALYSIS       ###\n")
cat("###   PARTS 2 & 3: DATABASE ARCHITECTURE, FEATURE ENGINEERING & 12 ML MODELS ###\n")
cat("###   AUTHORS: Karepalli Varun Reddy (24BDS1126) & Madheshwaran P (24BDS1159) ###\n")
cat("################################################################################\n\n")

start_pipeline_time <- Sys.time()

# Directory Resolution
if (file.exists("DA2_DA3/R/functions.R")) {
  base_dir <- "DA2_DA3"
  r_dir <- "DA2_DA3/R"
} else if (file.exists("R/functions.R")) {
  base_dir <- "."
  r_dir <- "R"
} else {
  base_dir <- "."
  r_dir <- "R"
}

# Source Core Functions & Verify Libraries
source(file.path(r_dir, "functions.R"))
ensure_packages()

pipeline_scripts <- c(
  "01_database.R",
  "02_data_audit.R",
  "03_feature_engineering.R",
  "04_feature_selection.R",
  "run_10_regression_models.R",
  "generate_regression_console_logs.R",
  "generate_regression_figures.R",
  "08_visualizations.R",
  "09_supplementary.R",
  "10_progress.R"
)

cat("--> Initializing master execution sequence (Scripts 01 through 10)...\n\n")

for (script in pipeline_scripts) {
  script_path <- file.path(r_dir, script)
  cat("======================================================================\n")
  cat("--> EXECUTING: ", script, "\n")
  cat("======================================================================\n")
  t0 <- Sys.time()
  source(script_path)
  t1 <- Sys.time()
  cat("--> Finished ", script, " in ", round(difftime(t1, t0, units = "secs"), 1), " seconds.\n\n")
}

total_elapsed <- round(difftime(Sys.time(), start_pipeline_time, units = "mins"), 2)

cat("################################################################################\n")
cat("###                   PIPELINE EXECUTION COMPLETE (ZERO ERRORS)              ###\n")
cat(sprintf("###   Total Execution Elapsed Time: %s minutes                          ###\n", total_elapsed))
cat("###   Deliverables Summary:                                                  ###\n")
cat("###   - SQLite Database: results/database/ai_jobs.db                         ###\n")
cat("###   - Presentation:   presentation/Final_Presentation.pptx (42 slides)     ###\n")
cat("###   - Report:         docs/Project_Report.docx (All 13 sections)           ###\n")
cat("###   - Execution Log:  docs/RStudio_Code_and_Output.docx                   ###\n")
cat("###   - Checklist:      screenshots/SCREENSHOT_CHECKLIST.md                  ###\n")
cat("################################################################################\n\n")
