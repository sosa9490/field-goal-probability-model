# Field Goal Success Probability Model

This project was developed as part of the LA Chargers Quantitative Analyst interview process. It estimates the difficulty of NFL field goal attempts using a kicker-agnostic model, then applies that model to real-world game scenarios and kicker evaluation.

The goal is to support coaching and decision-making with objective, data-driven analysis — whether for evaluating field goal range under specific game conditions or ranking kickers based on performance-adjusted metrics.

---

## 🏈 Project Objectives

1. **Model Development**  
   Build a model that predicts the baseline make probability of a field goal based on environmental and physical conditions.

2. **Scenario Analysis**  
   Apply the model to evaluate strategic decisions in specific game situations — e.g., whether to attempt a field goal or go for it on 4th down.

3. **Kicker Rankings**  
   Rank NFL kickers by "points added per attempt" — measuring how much each kicker outperforms or underperforms expectation after adjusting for kick difficulty.

---

## 📁 Repository Structure

```
.
├── analysis/                  # Scripts applying the model to game scenarios and kicker rankings
│   ├── Game_Scenario_One.R
│   ├── Game_Scenario_Two.R
│   └── Kicker_Rankings.R
├── model_dev/                # Core model building and selection
│   └── model_selection.R
├── models/                   # Final saved model objects
│   ├── log_model_smote_final.rds
│   └── platt_scaling_model.rds
├── scripts/                  # Supporting or production scripts
│   ├── build_pbp_field_descriptions.R
│   └── production_model.R
├── utilities/                # Reusable utility functions
│   ├── feature_engineering.R
│   ├── load_fg_data.R
│   └── helpers.R
├── data/                     # Source data used for modeling and evaluation
│   ├── field_descriptions.csv
│   └── field_descriptions_tagged_kicking.csv
├── deliverables/             # Final documents and outputs
│   ├── Model Explanation.pdf / docx
│   ├── Game Scenario Analysis.pdf / docx
│   └── kicker_rankings_2024.pdf
└── README.Rmd                # This file
```

---

## 🧠 Key Methodology

- **Model Type**: Logistic Regression with SMOTE for class imbalance
- **Calibration**: Platt scaling to improve probability estimates
- **Features**: Kick distance, weather (wind, temperature), turf type, dome vs. outdoor, game location
- **Data Source**: `nflfastR` play-by-play data + custom weather parsing

---

## 📊 Outputs

- `kicker_rankings_2024.pdf`: Evaluates each kicker’s performance above expectation.
- `Game Scenario Analysis.pdf`: Recommends go-for-it vs. field goal based on model-adjusted probabilities.
- `Model Explanation.pdf`: Details methodology, assumptions, and calibration.

---

## 🔧 How to Reproduce

1. **Install required packages**

```r
install.packages(c(
  "tidyverse", "nflreadr", "nflfastR",
  "caret", "themIs", "yardstick", "pROC", 
  "MLmetrics", "Metrics", "rsample", 
  "xgboost", "Matrix", "PRROC", 
  "gt", "webshot2", 
  "stringr", "readr", "glue", 
  "rmarkdown", "knitr", "janitor", "lubridate"
))
```

2. **Load and engineer features**

```r
source("utilities/load_fg_data.R")
source("utilities/feature_engineering.R")
```

3. **Train or load the model**

```r
# Option 1: Train from scratch
source("model_dev/model_selection.R")

# Option 2: Load final model
log_model <- readRDS("models/log_model_smote_final.rds")
platt_model <- readRDS("models/platt_scaling_model.rds")
```

4. **Run scenario or kicker analysis**

```r
source("analysis/Game_Scenario_One.R")
source("analysis/Game_Scenario_Two.R")
source("analysis/Kicker_Rankings.R")
```

---

## ✅ Deliverable Summary

| File                          | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `Model Explanation.pdf`       | Breakdown of model logic, performance, and purpose         |
| `Game Scenario Analysis.pdf`  | Recommendations for two key decision-making situations     |
| `kicker_rankings_2024.pdf`    | Evaluation of kicker performance above expected baseline   |

---

## 🙌 Acknowledgments

Built using `nflfastR`, `nflreadr`, and the R modeling ecosystem (`caret`, `yardstick`, `pROC`, `MLmetrics`, etc.).  
All modeling, engineering, and analysis by **Somak Sarkar**

