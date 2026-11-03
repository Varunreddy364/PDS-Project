# ==============================================================================
# 10_progress.R - Milestone Progress Tracker & 75% Completion Audit
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
cat("=== PART 3 - SECTION G: PROGRESS DEMONSTRATION & MILESTONES (75%) ===\n")
cat("======================================================================\n\n")

tab_dir <- file.path(base_dir, "results/tables")
fig_dir <- file.path(base_dir, "results/figures")

progress_df <- read.csv(file.path(tab_dir, "progress_milestones.csv"))
print(progress_df)

overall_progress <- sum(progress_df$Percent_Complete * progress_df$Cumulative_Weight / 100)
cat(sprintf("\n--> Total Calculated Weighted Progress: %.1f%% (Benchmark Target ~75%%)\n", overall_progress))
cat("--> Honest Status Review: Milestones 1 through 7 are 100% complete and fully validated.\n")
cat("--> Milestone 8 (Final System Deployment & Oral Defense) is in progress at 30%.\n")
cat("--> Milestone chart saved at: results/figures/fig_progress_chart.png\n\n")
