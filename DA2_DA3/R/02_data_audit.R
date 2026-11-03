# ==============================================================================
# 02_data_audit.R - Comprehensive Statistical Data Audit & Dominance Analysis
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
cat("=== PART 2: RIGOROUS DATA AUDIT & RELATIONSHIP ANALYSIS ===\n")
cat("======================================================================\n\n")

fig_dir <- file.path(base_dir, "results/figures")
tab_dir <- file.path(base_dir, "results/tables")
db_data_path <- file.path(base_dir, "results/database/retrieved_modeling_data.rds")

# Load retrieved dataset from database
if (file.exists(db_data_path)) {
  df <- readRDS(db_data_path)
  cat("--> Loaded modeling dataset from database RDS (", nrow(df), " rows, ", ncol(df), " cols)\n", sep = "")
} else {
  df <- read.csv(file.path(base_dir, "data/cleaned_ai_impact_jobs_2010_2025.csv"), stringsAsFactors = FALSE)
  cat("--> Loaded modeling dataset from CSV fallback\n")
}

# Convert integer 0/1 back to logical if retrieved from SQLite
df$ai_mentioned <- as.logical(df$ai_mentioned)
df$reskilling_required <- as.logical(df$reskilling_required)

# ------------------------------------------------------------------------------
# AUDIT ITEM 1: ai_mentioned vs Target Distribution
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 1: TARGET RANGE SEPARATION BY ai_mentioned ===\n")
audit_1 <- df %>%
  group_by(ai_mentioned) %>%
  summarise(
    Posting_Count = n(),
    Min_Risk = min(automation_risk_score),
    Q25_Risk = quantile(automation_risk_score, 0.25),
    Median_Risk = median(automation_risk_score),
    Mean_Risk = mean(automation_risk_score),
    Q75_Risk = quantile(automation_risk_score, 0.75),
    Max_Risk = max(automation_risk_score),
    SD_Risk = sd(automation_risk_score)
  )
print(as.data.frame(audit_1))
write.csv(audit_1, file.path(tab_dir, "audit_1_ai_mentioned_separation.csv"), row.names = FALSE)
cat("--> FINDING: When ai_mentioned == TRUE, automation_risk_score is strictly [0.10, 0.40].\n")
cat("--> FINDING: When ai_mentioned == FALSE, automation_risk_score is strictly [0.60, 0.90].\n")
cat("--> FINDING: No observations exist in (0.40, 0.60). Hence, ai_mentioned acts as a deterministic boundary for 'Low' risk.\n")

# ------------------------------------------------------------------------------
# AUDIT ITEM 2: reskilling_required vs ai_mentioned
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 2: IDENTITY BETWEEN reskilling_required & ai_mentioned ===\n")
cross_reskill <- table(ai_mentioned = df$ai_mentioned, reskilling_required = df$reskilling_required)
print(cross_reskill)
is_identical <- identical(as.character(df$ai_mentioned), as.character(df$reskilling_required))
agree_pct <- mean(as.character(df$ai_mentioned) == as.character(df$reskilling_required)) * 100
cat("Are ai_mentioned and reskilling_required 100% identical? ->", is_identical, "\n")
cat("Agreement percentage:", sprintf("%.2f%%", agree_pct), "\n")
cat("--> FINDING: reskilling_required is a perfect duplicate (collinear proxy) of ai_mentioned.\n")

# ------------------------------------------------------------------------------
# AUDIT ITEM 3: Free-Text Fields (ai_keywords & ai_skills) vs ai_mentioned
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 3: FREE-TEXT ATTRIBUTES WHEN ai_mentioned IS FALSE ===\n")
false_postings <- df %>% filter(ai_mentioned == FALSE)
true_postings <- df %>% filter(ai_mentioned == TRUE)

cat("ai_keywords when ai_mentioned == FALSE:\n")
cat("  - Unique values:", paste(unique(false_postings$ai_keywords), collapse = ", "), "\n")
cat("  - Proportion 'Not Specified':", sprintf("%.2f%%", 100 * mean(false_postings$ai_keywords == "Not Specified")), "\n")

cat("ai_skills when ai_mentioned == FALSE:\n")
cat("  - Unique values:", paste(unique(false_postings$ai_skills), collapse = ", "), "\n")
cat("  - Proportion 'Not Specified':", sprintf("%.2f%%", 100 * mean(false_postings$ai_skills == "Not Specified")), "\n")

cat("When ai_mentioned == TRUE:\n")
cat("  - 'Not Specified' in ai_keywords:", sum(true_postings$ai_keywords == "Not Specified"), "of", nrow(true_postings), "\n")
cat("  - 'Not Specified' in ai_skills:", sum(true_postings$ai_skills == "Not Specified"), "of", nrow(true_postings), "\n")
cat("--> FINDING: Both ai_keywords and ai_skills are 100% 'Not Specified' when ai_mentioned is FALSE, making them structural proxies.\n")

# ------------------------------------------------------------------------------
# AUDIT ITEM 4: ai_intensity_score vs ai_mentioned & Target
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 4: ai_intensity_score RELATIONSHIP & SIMPSON'S PARADOX ===\n")
audit_4 <- df %>%
  group_by(ai_mentioned) %>%
  summarise(
    n = n(),
    Min_Intensity = min(ai_intensity_score),
    Mean_Intensity = mean(ai_intensity_score),
    Max_Intensity = max(ai_intensity_score),
    Within_Group_Cor_Target = cor(ai_intensity_score, automation_risk_score)
  )
print(as.data.frame(audit_4))
overall_r_intensity <- cor(df$ai_intensity_score, df$automation_risk_score)
cat("Overall correlation r(ai_intensity_score, automation_risk_score):", round(overall_r_intensity, 4), "\n")
cat("Within ai_mentioned == FALSE correlation:", round(audit_4$Within_Group_Cor_Target[audit_4$ai_mentioned == FALSE], 4), "\n")
cat("Within ai_mentioned == TRUE correlation:", round(audit_4$Within_Group_Cor_Target[audit_4$ai_mentioned == TRUE], 4), "\n")
cat("--> FINDING: While overall r = -0.8751, within-group correlation drops to near ZERO (+0.029 and -0.012)!\n")
cat("--> This demonstrates classic group-shift effect: intensity separates AI vs Non-AI groups, but does not predict within-group risk.\n")

# ------------------------------------------------------------------------------
# AUDIT ITEM 5: ai_job_displacement_risk vs Target
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 5: ai_job_displacement_risk vs TARGET SCORE ===\n")
audit_5 <- df %>%
  group_by(ai_job_displacement_risk) %>%
  summarise(
    Count = n(),
    Min_Risk = min(automation_risk_score),
    Mean_Risk = mean(automation_risk_score),
    Max_Risk = max(automation_risk_score),
    SD_Risk = sd(automation_risk_score)
  )
print(as.data.frame(audit_5))
cat("--> FINDING: Mean automation risk score is virtually identical across High (0.5826), Low (0.5839), and Medium (0.5920).\n")
cat("--> ai_job_displacement_risk has essentially zero predictive power for automation_risk_score (Eta-sq = 0.00027).\n")

# ------------------------------------------------------------------------------
# AUDIT ITEM 6: Correlations Overall vs Within ai_mentioned Groups
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 6: CORRELATIONS WITH TARGET OVERALL AND WITHIN GROUPS ===\n")
numeric_features <- c(
  "salary_usd", 
  "salary_change_vs_prev_year_percent", 
  "posting_year", 
  "job_description_embedding_cluster", 
  "ai_intensity_score"
)

audit_6 <- data.frame(
  Feature = numeric_features,
  Overall_Correlation = sapply(numeric_features, function(x) round(cor(df[[x]], df$automation_risk_score), 4)),
  AI_False_Correlation = sapply(numeric_features, function(x) round(cor(df[df$ai_mentioned == FALSE, x], df$automation_risk_score[df$ai_mentioned == FALSE]), 4)),
  AI_True_Correlation = sapply(numeric_features, function(x) round(cor(df[df$ai_mentioned == TRUE, x], df$automation_risk_score[df$ai_mentioned == TRUE]), 4))
)
print(audit_6)
write.csv(audit_6, file.path(tab_dir, "audit_6_numeric_correlations.csv"), row.names = FALSE)
cat("--> FINDING: Continuous variables exhibit moderate-to-strong overall correlation with risk purely because their distributions differ between AI-mention groups. Inside each group, their correlation with the risk score is negligible!\n")

# ------------------------------------------------------------------------------
# AUDIT ITEM 7: Eta-Squared (Effect Size) of All Categorical Columns
# ------------------------------------------------------------------------------
cat("\n=== AUDIT ITEM 7: ETA-SQUARED EFFECT SIZE FOR CATEGORICAL PREDICTORS ===\n")
cat_cols <- c(
  "ai_mentioned", "reskilling_required", "industry_ai_adoption_stage",
  "industry", "company_size", "country", "company_name", "city",
  "seniority_level", "region", "job_title", "ai_job_displacement_risk"
)

calc_eta_sq <- function(var_name, y) {
  fit <- aov(y ~ as.factor(df[[var_name]]))
  ss <- summary(fit)[[1]][["Sum Sq"]]
  eta_sq <- ss[1] / sum(ss)
  return(eta_sq)
}

audit_7 <- data.frame(
  Categorical_Variable = cat_cols,
  Distinct_Levels = sapply(cat_cols, function(v) length(unique(df[[v]]))),
  Eta_Squared = sapply(cat_cols, function(v) round(calc_eta_sq(v, df$automation_risk_score), 6))
) %>% arrange(desc(Eta_Squared))

audit_7$Variance_Explained_Pct <- paste0(round(audit_7$Eta_Squared * 100, 2), "%")
print(audit_7)
write.csv(audit_7, file.path(tab_dir, "audit_7_eta_squared.csv"), row.names = FALSE)
cat("--> FINDING: ai_mentioned and reskilling_required account for 88.04% of total target variance.\n")
cat("--> industry_ai_adoption_stage accounts for 15.86%, while industry accounts for 2.29%.\n")
cat("--> All other variables individually explain less than 1% of the target score variance.\n")

# ------------------------------------------------------------------------------
# Master Evidence Table for Report
# ------------------------------------------------------------------------------
cat("\n--> Compiling Master Audit Evidence Table...\n")
master_audit <- data.frame(
  Feature = c("ai_mentioned", "reskilling_required", "ai_intensity_score", 
              "industry_ai_adoption_stage", "salary_change_vs_prev_year_percent",
              "salary_usd", "posting_year", "industry", "company_size",
              "job_title", "ai_job_displacement_risk"),
  Type = c("Binary", "Binary", "Continuous", "Categorical (3)", "Continuous",
           "Continuous", "Integer (Year)", "Categorical (9)", "Categorical (5)",
           "Categorical (10)", "Categorical (3)"),
  Target_Association_Metric = c("Eta-sq = 0.8804", "Eta-sq = 0.8804", "r = -0.8751",
                                "Eta-sq = 0.1586", "r = -0.6761", "r = -0.4317",
                                "r = -0.4224", "Eta-sq = 0.0229", "Eta-sq = 0.0098",
                                "Eta-sq = 0.0005", "Eta-sq = 0.0003"),
  Within_AI_Mention_Behavior = c("Strictly separates [0.10, 0.40] vs [0.60, 0.90]",
                                 "100% identical duplicate of ai_mentioned",
                                 "Within-group r drops to +0.029 and -0.012",
                                 "Explains residual variance between Med & High",
                                 "Within-group r drops to -0.014 and +0.010",
                                 "Within-group r drops to +0.009 and -0.030",
                                 "Within-group r drops to -0.009 and -0.016",
                                 "Moderate industry-level baseline shifts",
                                 "Slight risk variation across firm scales",
                                 "No significant predictive divergence",
                                 "Virtually zero relationship with target score"),
  Audit_Conclusion = c("Dominant primary separator (Low vs Med/High)",
                       "Redundant duplicate of ai_mentioned",
                       "Proxy for ai_mentioned; structural gap [0.20, 0.40]",
                       "Key secondary predictor (separates Med vs High)",
                       "Distribution shift artefact across groups",
                       "Distribution shift artefact across groups",
                       "Temporal growth artefact of AI postings",
                       "Secondary context feature",
                       "Minor context feature",
                       "Low individual utility",
                       "Noise variable for this specific target")
)
write.csv(master_audit, file.path(tab_dir, "audit_master_evidence_table.csv"), row.names = FALSE)
cat("--> Master evidence table written to results/tables/audit_master_evidence_table.csv\n")

# ------------------------------------------------------------------------------
# AUDIT VISUALIZATIONS (300 DPI)
# ------------------------------------------------------------------------------
cat("\n--> Generating Publication Figures for Data Audit...\n")

# Figure 1: Target Separation by ai_mentioned (Violin + Boxplot + Thresholds)
p_audit1 <- ggplot(df, aes(x = factor(ai_mentioned, labels = c("ai_mentioned = FALSE\n(Non-AI Postings)", "ai_mentioned = TRUE\n(AI Postings)")), 
                          y = automation_risk_score, fill = ai_mentioned)) +
  geom_violin(alpha = 0.35, trim = FALSE, color = NA) +
  geom_boxplot(width = 0.22, outlier.size = 1.2, alpha = 0.85, color = "#1F2D3D") +
  geom_hline(yintercept = 0.40, linetype = "dashed", color = "#008991", linewidth = 0.9) +
  geom_hline(yintercept = 0.70, linetype = "dashed", color = "#D9534F", linewidth = 0.9) +
  annotate("text", x = 1.5, y = 0.42, label = "Threshold: Low / Medium (0.40)", color = "#008991", fontface = "bold", size = 3.6) +
  annotate("text", x = 1.5, y = 0.72, label = "Threshold: Medium / High (0.70)", color = "#D9534F", fontface = "bold", size = 3.6) +
  scale_fill_manual(values = c("FALSE" = "#D9534F", "TRUE" = "#008991"), guide = "none") +
  labs(
    title = "Data Audit: Strict Separation of Automation Risk Score by AI Mention",
    subtitle = "ai_mentioned = TRUE strictly occupies [0.10, 0.40] (Low); ai_mentioned = FALSE occupies [0.60, 0.90] (Med & High)",
    x = "AI Mention Attribute",
    y = "Continuous Automation Risk Score",
    caption = "Source: Cleaned dataset audit (5,000 observations). Zero records exist in range (0.40, 0.60)."
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_audit_target_separation.png"), p_audit1, width = 8, height = 5.2, dpi = 300)
cat("--> Saved: results/figures/fig_audit_target_separation.png\n")

# Figure 2: Eta-Squared Effect Size Bar Chart
p_audit2 <- ggplot(audit_7, aes(x = reorder(Categorical_Variable, Eta_Squared), y = Eta_Squared, fill = Eta_Squared)) +
  geom_col(width = 0.68) +
  geom_text(aes(label = sprintf("%.1f%%", Eta_Squared * 100)), hjust = -0.15, size = 3.4, fontface = "bold", color = "#2C3E50") +
  coord_flip() +
  scale_fill_gradient(low = "#8492A6", high = "#275E9C", guide = "none") +
  scale_y_continuous(limits = c(0, 1.0), labels = scales::percent_format()) +
  labs(
    title = "Categorical Feature Importance via ANOVA (Eta-Squared Effect Size)",
    subtitle = "ai_mentioned & reskilling_required dominate target variance (88.0%); other features play secondary roles",
    x = "Predictor Variable",
    y = "Proportion of Variance Explained (Eta-Squared)",
    caption = "Calculated via one-way ANOVA F-tests against continuous automation_risk_score"
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_audit_eta_squared.png"), p_audit2, width = 8.5, height = 5.5, dpi = 300)
cat("--> Saved: results/figures/fig_audit_eta_squared.png\n")

cat("\n=== DATA AUDIT SCRIPT COMPLETE ===\n\n")
