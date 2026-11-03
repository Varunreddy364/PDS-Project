# Will AI Replace Your Job? 🤖
### Machine Learning-Based Analysis of Occupational AI Risk

[![Language](https://img.shields.io/badge/Language-R-blue)](https://www.r-project.org/)
[![Domain](https://img.shields.io/badge/Domain-Data%20Science-orange)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green)]()
[![Database](https://img.shields.io/badge/Database-SQLite-blue)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## 📌 Project Overview

Artificial Intelligence is transforming industries, automating workflows, and changing the skills required across different occupations. This raises an important question:

**Will AI replace your job?**

This project explores occupational automation risk using Exploratory Data Analysis (EDA), Statistical Inference, Feature Engineering, and Machine Learning.

Using the Global AI Job Market Dataset (2010–2025), the project analyses 5,000 job-posting records and develops multiple regression models to predict the recorded automation risk score based on job-related, salary, industry, company, geographical, and AI-related attributes.

The primary objective is to understand the factors associated with occupational automation risk and evaluate how effectively machine learning models can estimate risk scores.

## 🎯 Project Objectives

- Analyse the relationship between AI adoption and occupational automation risk.
- Explore salary, industry, experience, skills, and geographical patterns.
- Perform statistical analysis to identify significant relationships and group differences.
- Engineer meaningful predictive features using all 20 input variables.
- Develop and compare 10 regression algorithms.
- Optimize model performance through cross-validation and hyperparameter tuning.
- Evaluate model predictions using appropriate regression metrics.

## 📊 Dataset Description

**Dataset:** Global AI Job Market Dataset (2010–2025)

**Source:** [Kaggle – Terence Katua](https://www.kaggle.com/)

| Attribute | Details |
|---|---|
| Total Records | 5,000 |
| Original Variables | 22 |
| Input Variables | 20 |
| Target Variable | automation_risk_score |
| Time Period | 2010–2025 |
| Data Format | CSV |

The dataset contains information related to:

- Job titles and seniority levels
- Industry and company details
- Salary and salary changes
- AI mentions and AI intensity
- Core skills and AI-related skills
- Country, region, and city
- Industry AI adoption stage
- Automation and displacement risk indicators

**Target Variable:** `automation_risk_score`

The target is a continuous numerical variable representing the recorded automation risk score.

## 🛠️ Project Methodology

### 1. Dataset Collection and Exploratory Data Analysis

- Imported the dataset into RStudio.
- Examined dataset dimensions, structure, and summary statistics.
- Analysed salary distributions and job postings across industries and countries.
- Visualized AI adoption stages and salary variations across experience levels.
- Investigated the relationship between AI intensity and automation risk.

### 2. Data Preprocessing and Statistical Analysis

- Identified and handled missing text values.
- Replaced missing entries in `ai_keywords` and `ai_skills` with "Not Specified".
- Checked duplicate records and validated data integrity.
- Performed Pearson correlation analysis.
- Conducted group comparisons of automation risk based on AI mentions.

### 3. Feature Engineering and Database Integration

- Processed all 20 input variables.
- Extracted keyword counts and technical skill indicators.
- Created additional numerical and categorical features.
- Applied one-hot encoding to categorical variables.
- Expanded the feature matrix to 153 predictor columns.
- Integrated SQLite for structured data storage, retrieval, and SQL-based industry analysis.

### 4. Machine Learning Model Development

Developed and evaluated 10 regression algorithms:

1. Multiple Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Ridge Regression
5. Lasso Regression
6. Support Vector Regression (SVR)
7. K-Nearest Neighbors (KNN) Regression
8. Elastic-Net Regression
9. Gradient Boosting Regression
10. Neural Network Regression

An 80:20 training and testing strategy was used:

- Training records: 4,002 (80%)
- Testing records: 998 (20%)

### 5. Hyperparameter Tuning and Model Evaluation

- Applied 10-fold cross-validation.
- Optimized model-specific hyperparameters.
- Compared regression performance using MAE, RMSE, and R².
- Conducted feature importance analysis.
- Performed feature ablation experiments.
- Validated predictions against unseen test records.

## 📈 Model Performance

The best reported models achieved the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Lasso Regression | 0.0750 | 0.0867 | 87.82% |
| Elastic-Net Regression | 0.0750 | 0.0867 | 87.82% |
| Decision Tree Regression | 0.0750 | 0.0868 | 87.81% |
| Random Forest Regression | 0.0752 | 0.0874 | 87.62% |
| Gradient Boosting Regression | 0.0754 | 0.0871 | 87.70% |

**Best Reported Performance:**

Lasso and Elastic-Net Regression achieved the highest reported R² of **87.82%**, with an MAE of 0.0750 and RMSE of 0.0867.

These metrics were calculated on the held-out test dataset.

## 🔍 Additional Analysis

### Feature Importance Analysis

The project investigated the contribution of multiple occupational attributes, including:

- AI intensity score
- AI keyword counts
- Core skills and technical competencies
- Salary-related attributes
- Industry and seniority level

### Feature Ablation Analysis

An additional experiment evaluated model performance without direct AI intensity and AI keyword indicators.

The remaining non-AI occupational attributes achieved:

- R²: 86.8%
- MAE: 0.082

This demonstrates that traditional job-related attributes also contain predictive information about the recorded automation risk score.

### Prediction Validation

Individual high-risk and low-risk test records were examined by comparing actual automation risk scores with model predictions.

## 💻 Technologies and Tools Used

| Technology | Purpose |
|---|---|
| R Programming | Data analysis and machine learning |
| RStudio | Development environment |
| ggplot2 | Data visualization |
| dplyr | Data manipulation |
| tidyr | Data transformation |
| readr | CSV operations |
| SQLite | Database management |
| DBI | Database connectivity |
| rpart | Decision Tree Regression |
| randomForest / ranger | Random Forest |
| glmnet | Ridge, Lasso and Elastic-Net |
| kernlab | Support Vector Regression |
| kknn | K-Nearest Neighbors |
| gbm | Gradient Boosting |

## 📌 Key Findings

- AI intensity showed a strong negative correlation with automation risk in the dataset (Pearson r = -0.875).
- Salary changes also showed a negative correlation with automation risk (r = -0.676).
- Lasso and Elastic-Net achieved the best reported predictive performance.
- Decision Tree Regression delivered highly competitive results with relatively low training time.
- Non-AI occupational attributes alone retained substantial predictive performance.
- Multiple job-related factors contribute to the prediction of recorded automation risk.

## 🏁 Conclusion

This project demonstrates how statistical analysis and machine learning can be used to investigate occupational automation risk using job-market data.

The results highlight the importance of considering multiple factors, including AI exposure, salary, industry, seniority, and technical skills, rather than relying on a single attribute.

Although AI is transforming the nature of work, its impact varies across occupations. Understanding these patterns can help individuals and organizations make more informed decisions about career development, workforce planning, and reskilling.

**AI may transform jobs, but understanding its impact helps us prepare for the future.**

## 👨‍💻 Project Contributors

**Karepalli Varun Reddy**  
VIT Chennai | Computer Science and Engineering – Data Science

**Madheshwaran P**  
VIT Chennai | Computer Science and Engineering – Data Science

## 📚 Academic Project

Developed as part of the **Programming for Data Science** coursework at Vellore Institute of Technology, Chennai.

---

⭐ If you find this project interesting, feel free to explore the repository and its analysis.
