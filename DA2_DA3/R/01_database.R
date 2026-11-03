# ==============================================================================
# 01_database.R - Database Connectivity, Schema Management, and SQL Analysis
# Authors: Karepalli Varun Reddy (24BDS1126) & Madheshwaran P (24BDS1159)
# Course: Programming for Data Science, VIT Chennai
# ==============================================================================

# Ensure working directory compatibility
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
cat("=== PART 2 - SECTION A: DATABASE CONNECTIVITY & SQL RETRIEVAL ===\n")
cat("======================================================================\n\n")

# Path definitions
db_path <- file.path(base_dir, "results/database/ai_jobs.db")
csv_path <- file.path(base_dir, "data/cleaned_ai_impact_jobs_2010_2025.csv")
fig_dir <- file.path(base_dir, "results/figures")
tab_dir <- file.path(base_dir, "results/tables")

dir.create(dirname(db_path), recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Establish SQLite Connection
cat("--> Connecting to SQLite database at:", db_path, "\n")
con <- dbConnect(RSQLite::SQLite(), dbname = db_path)

# 2. Ingest Cleaned Dataset into Table 'job_postings'
cat("--> Reading cleaned dataset from:", csv_path, "\n")
raw_data <- read.csv(csv_path, stringsAsFactors = FALSE)
cat("--> Writing 'job_postings' table (5,000 rows, 22 columns)...\n")
dbWriteTable(con, "job_postings", raw_data, overwrite = TRUE)

# 3. Create Second Lookup Table: 'industry_summary'
cat("--> Creating lookup table 'industry_summary'...\n")
sql_create_lookup <- "
CREATE TABLE IF NOT EXISTS industry_summary AS
SELECT 
    industry,
    COUNT(*) AS total_postings,
    ROUND(AVG(salary_usd), 2) AS avg_industry_salary,
    ROUND(AVG(automation_risk_score), 4) AS avg_industry_risk,
    ROUND(AVG(ai_intensity_score), 4) AS avg_industry_ai_intensity
FROM job_postings
GROUP BY industry;
"
dbExecute(con, "DROP TABLE IF EXISTS industry_summary;")
dbExecute(con, sql_create_lookup)

# 4. Create View: 'v_high_risk_postings'
cat("--> Creating view 'v_high_risk_postings'...\n")
sql_create_view <- "
CREATE VIEW IF NOT EXISTS v_high_risk_postings AS
SELECT 
    job_id,
    job_title,
    industry,
    company_size,
    salary_usd,
    automation_risk_score,
    ai_mentioned
FROM job_postings
WHERE automation_risk_score > 0.70;
"
dbExecute(con, "DROP VIEW IF EXISTS v_high_risk_postings;")
dbExecute(con, sql_create_view)

# 5. Schema Introspection
cat("\n=== DATABASE SCHEMA INTROSPECTION ===\n")
tables <- dbListTables(con)
cat("Tables and Views present in database:\n")
print(tables)

fields <- dbListFields(con, "job_postings")
cat("\nFields in 'job_postings' (", length(fields), " columns):\n", sep = "")
print(fields)

row_count <- dbGetQuery(con, "SELECT COUNT(*) AS total_rows FROM job_postings;")
cat("\nTotal records in 'job_postings':", row_count$total_rows, "\n")

cat("\nHead of 'job_postings' (first 5 records):\n")
head_data <- dbGetQuery(con, "SELECT job_id, job_title, industry, salary_usd, automation_risk_score FROM job_postings LIMIT 5;")
print(head_data)

# 6. Execute Required 6 SQL Queries with Code and Output Display
cat("\n======================================================================\n")
cat("=== EXECUTING 6 REQUIRED SQL ANALYTICAL QUERIES ===\n")
cat("======================================================================\n\n")

# QUERY 1: Industry-level Aggregations with HAVING
sql_q1 <- "
SELECT 
    industry,
    COUNT(*) AS posting_count,
    ROUND(AVG(automation_risk_score), 4) AS avg_risk_score,
    ROUND(AVG(salary_usd), 2) AS avg_salary_usd,
    ROUND(MIN(salary_usd), 2) AS min_salary,
    ROUND(MAX(salary_usd), 2) AS max_salary
FROM job_postings
GROUP BY industry
HAVING COUNT(*) >= 300
ORDER BY avg_risk_score DESC;
"
cat("=== QUERY 1: Industry Count, Average Risk, and Salary with HAVING ===\n")
cat("SQL Command:\n", sql_q1, "\n")
res_q1 <- dbGetQuery(con, sql_q1)
print(res_q1)
write.csv(res_q1, file.path(tab_dir, "sql_query1_industry_risk.csv"), row.names = FALSE)

# QUERY 2: Risk by Seniority Level and Company Size
sql_q2 <- "
SELECT 
    seniority_level,
    company_size,
    COUNT(*) AS count,
    ROUND(AVG(automation_risk_score), 4) AS avg_risk,
    ROUND(AVG(salary_usd), 2) AS avg_salary
FROM job_postings
GROUP BY seniority_level, company_size
ORDER BY seniority_level, avg_risk DESC;
"
cat("\n=== QUERY 2: Automation Risk by Seniority Level and Company Size ===\n")
cat("SQL Command:\n", sql_q2, "\n")
res_q2 <- dbGetQuery(con, sql_q2)
print(head(res_q2, 10))
cat("... (", nrow(res_q2), " rows total)\n")
write.csv(res_q2, file.path(tab_dir, "sql_query2_seniority_company_size.csv"), row.names = FALSE)

# QUERY 3: Yearly Trend of Risk and Salary
sql_q3 <- "
SELECT 
    posting_year,
    COUNT(*) AS total_postings,
    ROUND(AVG(automation_risk_score), 4) AS avg_risk_score,
    ROUND(AVG(salary_usd), 2) AS avg_salary_usd,
    ROUND(SUM(CASE WHEN ai_mentioned = 1 OR ai_mentioned = 'TRUE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ai_mention_pct
FROM job_postings
GROUP BY posting_year
ORDER BY posting_year ASC;
"
cat("\n=== QUERY 3: Yearly Temporal Trend of Automation Risk and Salary ===\n")
cat("SQL Command:\n", sql_q3, "\n")
res_q3 <- dbGetQuery(con, sql_q3)
print(res_q3)
write.csv(res_q3, file.path(tab_dir, "sql_query3_yearly_trends.csv"), row.names = FALSE)

# QUERY 4: Top 10 and Bottom 10 Job Titles by Risk with Minimum Count
sql_q4_top <- "
SELECT 
    job_title,
    COUNT(*) AS job_count,
    ROUND(AVG(automation_risk_score), 4) AS avg_risk,
    ROUND(AVG(salary_usd), 2) AS avg_salary
FROM job_postings
GROUP BY job_title
HAVING COUNT(*) >= 20
ORDER BY avg_risk DESC
LIMIT 5;
"
sql_q4_bot <- "
SELECT 
    job_title,
    COUNT(*) AS job_count,
    ROUND(AVG(automation_risk_score), 4) AS avg_risk,
    ROUND(AVG(salary_usd), 2) AS avg_salary
FROM job_postings
GROUP BY job_title
HAVING COUNT(*) >= 20
ORDER BY avg_risk ASC
LIMIT 5;
"
cat("\n=== QUERY 4: Extreme Risk Profiles by Job Title (Top 5 & Bottom 5) ===\n")
res_q4_top <- dbGetQuery(con, sql_q4_top)
res_q4_bot <- dbGetQuery(con, sql_q4_bot)
cat("Highest Risk Job Titles:\n")
print(res_q4_top)
cat("\nLowest Risk Job Titles:\n")
print(res_q4_bot)
res_q4 <- rbind(cbind(Category = "Highest Risk", res_q4_top),
                cbind(Category = "Lowest Risk", res_q4_bot))
write.csv(res_q4, file.path(tab_dir, "sql_query4_job_titles_risk.csv"), row.names = FALSE)

# QUERY 5: CTE Comparing Each Industry to Global Benchmark
sql_q5 <- "
WITH Benchmark AS (
    SELECT 
        AVG(automation_risk_score) AS global_avg_risk,
        AVG(salary_usd) AS global_avg_salary
    FROM job_postings
)
SELECT 
    j.industry,
    COUNT(*) AS n_postings,
    ROUND(AVG(j.automation_risk_score), 4) AS industry_avg_risk,
    ROUND(b.global_avg_risk, 4) AS benchmark_risk,
    ROUND(AVG(j.automation_risk_score) - b.global_avg_risk, 4) AS risk_deviation,
    ROUND(AVG(j.salary_usd), 2) AS industry_avg_salary,
    ROUND(b.global_avg_salary, 2) AS benchmark_salary,
    ROUND(AVG(j.salary_usd) - b.global_avg_salary, 2) AS salary_deviation
FROM job_postings j
CROSS JOIN Benchmark b
GROUP BY j.industry
ORDER BY risk_deviation DESC;
"
cat("\n=== QUERY 5: CTE Benchmark Analysis (Industry vs Overall Mean) ===\n")
cat("SQL Command:\n", sql_q5, "\n")
res_q5 <- dbGetQuery(con, sql_q5)
print(res_q5)
write.csv(res_q5, file.path(tab_dir, "sql_query5_cte_industry_benchmark.csv"), row.names = FALSE)

# QUERY 6: Relational JOIN between job_postings and lookup table
sql_q6 <- "
SELECT 
    j.job_id,
    j.job_title,
    j.industry,
    j.salary_usd,
    s.avg_industry_salary,
    ROUND(j.salary_usd - s.avg_industry_salary, 2) AS salary_premium,
    j.automation_risk_score,
    s.avg_industry_risk
FROM job_postings j
INNER JOIN industry_summary s ON j.industry = s.industry
ORDER BY j.job_id ASC
LIMIT 10;
"
cat("\n=== QUERY 6: Relational INNER JOIN with 'industry_summary' Lookup ===\n")
cat("SQL Command:\n", sql_q6, "\n")
res_q6 <- dbGetQuery(con, sql_q6)
print(res_q6)
write.csv(res_q6, file.path(tab_dir, "sql_query6_relational_join.csv"), row.names = FALSE)

# 7. Chart at least two SQL Query Results
cat("\n--> Generating Figures for SQL Query Results (300 DPI)...\n")

# Figure 1: Industry Risk vs Benchmark (Query 5)
p_sql1 <- ggplot(res_q5, aes(x = reorder(industry, risk_deviation), y = risk_deviation, fill = risk_deviation > 0)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#2C3E50", linewidth = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#D9534F", "FALSE" = "#008991")) +
  labs(
    title = "Industry Automation Risk Deviation from Global Benchmark (SQL CTE)",
    subtitle = "Positive deviation indicates higher risk than overall benchmark (0.5862)",
    x = "Industry Sector",
    y = "Risk Score Deviation (Industry Mean - Global Mean)",
    caption = "Source: SQL Query 5 CTE on SQLite database results/database/ai_jobs.db"
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_sql_industry_benchmark.png"), p_sql1, width = 8, height = 5, dpi = 300)
cat("--> Saved: results/figures/fig_sql_industry_benchmark.png\n")

# Figure 2: Yearly Trends of Risk and AI Mentions (Query 3)
p_sql2 <- ggplot(res_q3, aes(x = posting_year)) +
  geom_line(aes(y = avg_risk_score, color = "Automation Risk Score"), linewidth = 1.2) +
  geom_point(aes(y = avg_risk_score, color = "Automation Risk Score"), size = 3) +
  geom_line(aes(y = ai_mention_pct / 100, color = "AI Mention Proportion"), linewidth = 1.2, linetype = "dotdash") +
  geom_point(aes(y = ai_mention_pct / 100, color = "AI Mention Proportion"), size = 3) +
  scale_y_continuous(
    name = "Average Automation Risk Score (0.10 - 0.90)",
    sec.axis = sec_axis(~ . * 100, name = "AI Mention Proportion (%)")
  ) +
  scale_color_manual(
    name = "Metric",
    values = c("Automation Risk Score" = "#D9534F", "AI Mention Proportion" = "#008991")
  ) +
  labs(
    title = "Yearly Temporal Trends: Automation Risk vs AI Job Mentions (2010 - 2025)",
    subtitle = "Rising AI mentions strongly correlate with a secular decline in aggregate automation risk",
    x = "Job Posting Year",
    caption = "Source: SQL Query 3 aggregation on SQLite database results/database/ai_jobs.db"
  ) +
  theme_pds()

ggsave(file.path(fig_dir, "fig_sql_yearly_trends.png"), p_sql2, width = 8, height = 5, dpi = 300)
cat("--> Saved: results/figures/fig_sql_yearly_trends.png\n")

# 8. Retrieve Modeling Dataset from Database for Downstream ML
cat("\n--> Retrieving pristine modeling dataset via SQL SELECT * FROM job_postings...\n")
modeling_data <- dbGetQuery(con, "SELECT * FROM job_postings;")
cat("--> Successfully retrieved", nrow(modeling_data), "rows and", ncol(modeling_data), "columns.\n")

# Save retrieved dataset to RDS for downstream scripts
saveRDS(modeling_data, file.path(base_dir, "results/database/retrieved_modeling_data.rds"))

# Clean Disconnect
dbDisconnect(con)
cat("--> Database connection closed cleanly.\n")
cat("=== SECTION A COMPLETE ===\n\n")
