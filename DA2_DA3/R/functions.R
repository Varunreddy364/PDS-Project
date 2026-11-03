# ==============================================================================
# functions.R - Core Helper Functions for DA2 & DA3 Machine Learning Pipeline
# Authors: Karepalli Varun Reddy (24BDS1126) & Madheshwaran P (24BDS1159)
# Course: Programming for Data Science, VIT Chennai
# ==============================================================================

# Suppress annoying warnings
options(warn = -1)

# 1. Package Management & Loader
ensure_packages <- function() {
  required_pkgs <- c(
    "tidyverse", "janitor", "DBI", "RSQLite", "caret", "MASS", "e1071",
    "naivebayes", "kknn", "rpart", "rpart.plot", "ranger", "xgboost",
    "gbm", "glmnet", "C50", "ipred", "nnet", "pROC", "PRROC",
    "MLmetrics", "doParallel", "patchwork", "corrplot", "flextable", "officer"
  )
  
  for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
    }
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
  invisible(TRUE)
}

# 2. Plotting Theme
theme_pds <- function(base_size = 11, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2, color = "#1F2D3D", margin = margin(b = 6)),
      plot.subtitle = element_text(size = base_size - 1, color = "#5A6B7C", margin = margin(b = 8)),
      plot.caption = element_text(size = base_size - 2, color = "#8492A6", hjust = 1, margin = margin(t = 6)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#EBF0F5", linewidth = 0.5),
      panel.background = element_rect(fill = "#FAFCFD", color = NA),
      plot.background = element_rect(fill = "#FFFFFF", color = NA),
      axis.title = element_text(face = "bold", size = base_size, color = "#2C3E50"),
      axis.text = element_text(size = base_size - 1, color = "#47525E"),
      legend.title = element_text(face = "bold", size = base_size - 1, color = "#2C3E50"),
      legend.text = element_text(size = base_size - 1, color = "#47525E"),
      legend.position = "bottom",
      strip.background = element_rect(fill = "#E8EEF5", color = NA),
      strip.text = element_text(face = "bold", color = "#1F2D3D", size = base_size)
    )
}

pds_colors <- c(
  "Low" = "#008991",       # Deep Teal
  "Medium" = "#E6AB37",    # Ochre Gold
  "High" = "#D9534F",      # Warm Coral / Red
  "Primary" = "#275E9C",   # Navy Blue
  "Secondary" = "#008991", # Teal
  "Accent" = "#E6AB37",    # Gold
  "Neutral" = "#2C3E50"    # Charcoal
)

# 3. Feature Engineering Function
engineer_features <- function(df) {
  # 1. Target Multi-Class Definition
  if ("automation_risk_score" %in% colnames(df)) {
    df$risk_category <- cut(
      df$automation_risk_score,
      breaks = c(-Inf, 0.40, 0.70, Inf),
      labels = c("Low", "Medium", "High"),
      right = TRUE
    )
    df$risk_category <- factor(df$risk_category, levels = c("Low", "Medium", "High"))
  }
  
  # 2. Text Parsing: Counts, Not-Specified Flags, and Binary Presence
  # Helper to parse binary match
  has_term <- function(text_col, term) {
    as.integer(grepl(term, text_col, ignore.case = TRUE))
  }
  
  # Helper to count comma-separated items
  count_items <- function(text_col) {
    sapply(as.character(text_col), function(x) {
      if (is.na(x) || x == "" || trimws(x) == "Not Specified") return(0L)
      parts <- strsplit(x, "[,;|]")[[1]]
      parts <- trimws(parts)
      length(parts[parts != "" & parts != "Not Specified"])
    }, USE.NAMES = FALSE)
  }
  
  # ai_keywords features
  df$ai_keywords_count <- count_items(df$ai_keywords)
  df$ai_keywords_not_specified <- as.integer(grepl("Not Specified", df$ai_keywords, ignore.case = TRUE) | df$ai_keywords == "")
  df$kw_nlp <- has_term(df$ai_keywords, "\\bNLP\\b")
  df$kw_reinforcement_learning <- has_term(df$ai_keywords, "reinforcement learning")
  df$kw_deep_learning <- has_term(df$ai_keywords, "deep learning")
  df$kw_mlops <- has_term(df$ai_keywords, "MLOps")
  df$kw_computer_vision <- has_term(df$ai_keywords, "computer vision")
  df$kw_machine_learning <- has_term(df$ai_keywords, "machine learning")
  df$kw_generative_ai <- has_term(df$ai_keywords, "generative AI")
  df$kw_llms <- has_term(df$ai_keywords, "LLMs")
  
  # core_skills features
  df$core_skills_count <- count_items(df$core_skills)
  df$core_skills_not_specified <- as.integer(grepl("Not Specified", df$core_skills, ignore.case = TRUE) | df$core_skills == "")
  df$skill_business_analysis <- has_term(df$core_skills, "Business Analysis")
  df$skill_sql <- has_term(df$core_skills, "\\bSQL\\b")
  df$skill_python <- has_term(df$core_skills, "Python")
  df$skill_software_eng <- has_term(df$core_skills, "Software Engineering")
  df$skill_cloud_computing <- has_term(df$core_skills, "Cloud Computing")
  df$skill_research <- has_term(df$core_skills, "Research")
  df$skill_data_analysis <- has_term(df$core_skills, "Data Analysis")
  df$skill_project_management <- has_term(df$core_skills, "Project Management")
  df$skill_communication <- has_term(df$core_skills, "Communication")
  df$skill_statistics <- has_term(df$core_skills, "Statistics")
  
  # ai_skills features
  df$ai_skills_count <- count_items(df$ai_skills)
  df$ai_skills_not_specified <- as.integer(grepl("Not Specified", df$ai_skills, ignore.case = TRUE) | df$ai_skills == "")
  df$aiskill_nlp <- has_term(df$ai_skills, "\\bNLP\\b")
  df$aiskill_reinforcement_learning <- has_term(df$ai_skills, "reinforcement learning")
  df$aiskill_deep_learning <- has_term(df$ai_skills, "deep learning")
  df$aiskill_mlops <- has_term(df$ai_skills, "MLOps")
  df$aiskill_computer_vision <- has_term(df$ai_skills, "computer vision")
  df$aiskill_machine_learning <- has_term(df$ai_skills, "machine learning")
  df$aiskill_generative_ai <- has_term(df$ai_skills, "generative AI")
  df$aiskill_llms <- has_term(df$ai_skills, "LLMs")
  
  # 3. Numeric & Ordered Features
  df$log_salary <- log(pmax(df$salary_usd, 1))
  df$salary_rank_industry <- ave(df$salary_usd, df$industry, FUN = function(x) rank(x, ties.method = "average") / length(x))
  df$years_since_2010 <- df$posting_year - 2010
  df$salary_increased <- as.integer(df$salary_change_vs_prev_year_percent > 0)
  
  # Seniority level ordered integer
  seniority_map <- c("Entry" = 1, "Junior" = 2, "Mid" = 3, "Senior" = 4, "Lead" = 5, "Executive" = 6)
  df$seniority_numeric <- seniority_map[as.character(df$seniority_level)]
  df$seniority_numeric[is.na(df$seniority_numeric)] <- 3
  
  # Binary indicators for boolean columns
  df$ai_mentioned_num <- as.integer(df$ai_mentioned)
  df$reskilling_num <- as.integer(df$reskilling_required)
  
  # Convert categorical variables to clean factors
  cat_vars <- c("country", "region", "city", "company_name", "company_size",
                "industry", "job_title", "seniority_level", "ai_mentioned",
                "reskilling_required", "ai_job_displacement_risk",
                "industry_ai_adoption_stage")
  for (cv in cat_vars) {
    if (cv %in% colnames(df)) {
      df[[cv]] <- as.factor(df[[cv]])
    }
  }
  
  return(df)
}

# 4. Multi-class Metric Evaluation
compute_multiclass_metrics <- function(y_true, y_pred, prob_mat = NULL, train_time_sec = 0) {
  classes <- levels(y_true)
  n_classes <- length(classes)
  cm <- table(Actual = y_true, Predicted = y_pred)
  
  # Overall Accuracy
  accuracy <- sum(diag(cm)) / sum(cm)
  
  # Per-class Sensitivity (Recall), Specificity, Precision
  recall_per_class <- numeric(n_classes)
  precision_per_class <- numeric(n_classes)
  f1_per_class <- numeric(n_classes)
  balanced_acc_per_class <- numeric(n_classes)
  
  for (i in 1:n_classes) {
    tp <- cm[i, i]
    fn <- sum(cm[i, ]) - tp
    fp <- sum(cm[, i]) - tp
    tn <- sum(cm) - tp - fn - fp
    
    sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
    prec <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
    f1 <- ifelse((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0)
    
    recall_per_class[i] <- sens
    precision_per_class[i] <- prec
    f1_per_class[i] <- f1
    balanced_acc_per_class[i] <- (sens + spec) / 2
  }
  
  macro_recall <- mean(recall_per_class)
  macro_precision <- mean(precision_per_class)
  macro_f1 <- mean(f1_per_class)
  balanced_accuracy <- mean(balanced_acc_per_class)
  
  # Cohen's Kappa
  p_o <- accuracy
  p_e <- sum(rowSums(cm) * colSums(cm)) / (sum(cm)^2)
  kappa <- ifelse((1 - p_e) != 0, (p_o - p_e) / (1 - p_e), 0)
  
  # Matthews Correlation Coefficient (Multiclass generalization - Gorodkin 2004)
  cov_xy <- sum(diag(cm)) * sum(cm) - sum(rowSums(cm) * colSums(cm))
  var_x <- sum(cm)^2 - sum(rowSums(cm)^2)
  var_y <- sum(cm)^2 - sum(colSums(cm)^2)
  mcc <- ifelse(var_x > 0 && var_y > 0, cov_xy / sqrt(var_x * var_y), 0)
  
  # Multi-class Log Loss & Macro AUC
  log_loss <- NA_real_
  macro_auc <- NA_real_
  
  if (!is.null(prob_mat)) {
    # Ensure column names match classes
    prob_mat <- as.matrix(prob_mat)
    if (is.null(colnames(prob_mat))) {
      colnames(prob_mat) <- classes
    }
    # Normalize probabilities to sum to 1
    prob_mat <- t(apply(prob_mat, 1, function(p) {
      s <- sum(p)
      if (s > 0) p / s else rep(1 / n_classes, n_classes)
    }))
    
    # Log loss
    eps <- 1e-15
    prob_mat_clamped <- pmin(pmax(prob_mat, eps), 1 - eps)
    y_mat <- model.matrix(~ y_true - 1)
    log_loss <- -sum(y_mat * log(prob_mat_clamped)) / length(y_true)
    
    # Macro AUC (One-vs-Rest average)
    auc_list <- numeric(n_classes)
    for (i in 1:n_classes) {
      bin_y <- as.numeric(y_true == classes[i])
      pred_prob <- prob_mat[, classes[i]]
      if (length(unique(bin_y)) > 1) {
        roc_obj <- suppressMessages(pROC::roc(bin_y, pred_prob, quiet = TRUE, direction = "<"))
        auc_list[i] <- as.numeric(pROC::auc(roc_obj))
      } else {
        auc_list[i] <- 0.5
      }
    }
    macro_auc <- mean(auc_list)
  }
  
  data.frame(
    Accuracy = round(accuracy, 4),
    Balanced_Accuracy = round(balanced_accuracy, 4),
    Kappa = round(kappa, 4),
    Macro_Precision = round(macro_precision, 4),
    Macro_Recall = round(macro_recall, 4),
    Macro_F1 = round(macro_f1, 4),
    Macro_AUC = round(macro_auc, 4),
    Log_Loss = round(log_loss, 4),
    MCC = round(mcc, 4),
    Training_Time_Sec = round(train_time_sec, 2)
  )
}

# 5. Database Helper
get_db_connection <- function(db_path = "DA2_DA3/results/database/ai_jobs.db") {
  dir.create(dirname(db_path), recursive = TRUE, showWarnings = FALSE)
  DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
}

cat("[functions.R] Helper functions loaded successfully.\n")
