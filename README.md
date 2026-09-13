# 🚀 Smart Outcome Predictor

### Ensemble Learning for Student Performance — Classification + Regression

<p align="center">
  <b>Bagging</b> •
  <b>AdaBoost</b> •
  <b>Gradient Boosting</b> •
  <b>LightGBM</b> •
  <b>XGBoost</b> •
  <b>Voting</b> •
  <b>Stacking</b>
</p>



<p align="center">
  <b>Bagging</b> •
  <b>AdaBoost</b> •
  <b>Gradient Boosting</b> •
  <b>LightGBM</b> •
  <b>XGBoost</b> •
  <b>Voting</b> •
  <b>Stacking</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-Boosting-9ACD32">
  <img src="https://img.shields.io/badge/XGBoost-Boosting-EC4E20">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
</p>

</div>

---

## 📌 Project Overview

**Smart Outcome Predictor** is a Machine Learning project based on **Ensemble Learning**.

The project performs two predictive tasks:

| Task | Target | Objective |
|---|---|---|
| 🎯 Classification | `completion_status` | Predict whether a student completes the course |
| 📈 Regression | `final_score` | Predict the student's final score |

The project compares different ensemble learning techniques and evaluates their performance using suitable metrics.

---

## 🎯 Project Objectives

- Understand Ensemble Learning.
- Compare Bagging, Boosting, Voting and Stacking.
- Build classification and regression models.
- Compare a single Decision Tree with Bagging.
- Implement AdaBoost.
- Implement Gradient Boosting.
- Implement LightGBM.
- Implement XGBoost.
- Compare Hard Voting and Soft Voting.
- Build Stacking Classifier and Stacking Regressor.
- Evaluate and select the best-performing models.

---
🔍 Important Features

-age
-country_region
-device_type
-education_background
- course_level
- course_category
- course_start_date
- week_of_year
- sessions
- time_spent_hours
- videos_watched
- quiz_attempts
- assignments_submitted
- forum_posts
- avg_quiz_score
- attendance_rate
  

## 🎯 Target Variables

### 1. `completion_status`

Used for **Classification**.

The model predicts whether a student will complete the course.

### 2. `final_score`

Used for **Regression**.

The model predicts the student's final score.


### 📊 Important Features

The dataset contains the following important features:

- `age`
- `country_region`
- `device_type`
- `education_background`
- `course_level`
- `course_category`
- `course_start_date`
- `week_of_year`
- `sessions`
- `time_spent_hours`
- `videos_watched`
- `quiz_attempts`
- `assignments_submitted`
- `forum_posts`
- `avg_quiz_score`
- `attendance_rate`

> **Note:** The `student_id` column is treated as an identifier and is not used as a predictive feature.
>
> # 🧹 Data Preprocessing

The project uses **ColumnTransformer** and preprocessing pipelines.
## 🔢 Numerical Features

```text
Missing Values
      ↓
Median Imputation
      ↓
StandardScaler
```

---
## 🔤 Categorical Features

```text
Missing Values
      ↓
Most Frequent Imputation
      ↓
OneHotEncoder
```

This preprocessing converts numerical and categorical data into a suitable format for machine learning models.

# ✂️ Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

### 🏋️ Training Data

Used to train machine learning models.

### 🧪 Testing Data

Used to evaluate the performance of trained models on unseen data.

# 🧠 Machine Learning Models

## 1️⃣ Bagging

**Bagging (Bootstrap Aggregating)** trains multiple models on different bootstrap samples and combines their predictions.

### Implemented Models

```text
BaggingClassifier
BaggingRegressor
```

### 🌟 Main Benefits

Bagging helps to:

- 🔹 Reduce variance
- 🔹 Improve stability
- 🔹 Reduce overfitting
- 🔹 Improve prediction performance

- ## 2️⃣ AdaBoost

**AdaBoost (Adaptive Boosting)** trains weak learners sequentially.

Each new learner focuses more on the observations that were incorrectly predicted by previous learners.

### 📌 Implemented Models

```text
AdaBoostClassifier
AdaBoostRegressor
```

### 🌟 Main Benefit

AdaBoost improves weak learners by concentrating on previous errors.

---

## 3️⃣ Gradient Boosting

**Gradient Boosting** builds models sequentially to reduce prediction errors.

### 📌 Implemented Models

```text
GradientBoostingClassifier
GradientBoostingRegressor
```

The project also compares different:

```text
Learning Rates
Number of Estimators
```

---

## 4️⃣ LightGBM

**LightGBM (Light Gradient Boosting Machine)** is an efficient gradient boosting algorithm.

### 📌 Implemented Models

```text
LGBMClassifier
LGBMRegressor
```

### 🌟 Advantages

- ⚡ Fast training
- 💾 Efficient memory usage
- 📊 Good predictive performance
- 📈 Suitable for large datasets

---

## 5️⃣ XGBoost

**XGBoost (Extreme Gradient Boosting)** is a powerful and optimized boosting algorithm.

### 📌 Implemented Models

```text
XGBClassifier
XGBRegressor
```

### 🌟 Advantages

- 🚀 High performance
- 🛡️ Regularization
- 🔗 Handles complex relationships
- 💪 Powerful boosting algorithm

- # 🗳️ Voting Ensemble

Voting combines predictions from multiple classification models.

## 🟢 Hard Voting

Hard Voting selects the class that receives the majority of votes.

### 📌 Example

```text
Model 1 → Class 1
Model 2 → Class 0
Model 3 → Class 1

Final Prediction → Class 1
```

---

## 🔵 Soft Voting

Soft Voting combines prediction probabilities from different classification models.

The project compares:

```text
Hard Voting
Soft Voting
```

to determine which performs better.

---

# 🧩 Stacking Ensemble

**Stacking** combines multiple base models and uses a final **meta-model** to make the final prediction.

## 🎯 Stacking Classifier

```text
Base Models
     ↓
Predictions
     ↓
Meta Model
     ↓
Final Classification
```

## 📈 Stacking Regressor

```text
Base Models
     ↓
Predictions
     ↓
Meta Model
     ↓
Final Score Prediction
```

---

# 📊 Evaluation Metrics

## 🎯 Classification Metrics

### Accuracy

Measures the percentage of correctly predicted observations.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1 Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the ability of a classification model to distinguish between classes.

---

## 📈 Regression Metrics

### MAE

**Mean Absolute Error** measures the average absolute difference between actual and predicted values.

### MSE

**Mean Squared Error** measures the average squared prediction error.

### RMSE

**Root Mean Squared Error** gives more importance to larger prediction errors.

### R² Score

Measures how much variation in the target variable is explained by the model.

---

# 🏆 Final Model Performance

## 🎯 Best Classification Model

### 🥇 Stacking Classifier

| Metric | Score |
|---|---:|
| Accuracy | **74.42%** |
| Precision | **0.6867** |
| Recall | **0.5846** |
| F1 Score | **0.6316** |
| ROC-AUC | **0.7940** |

### ✅ Result

**Stacking Classifier achieved the strongest overall classification performance.**

---

# 📈 Best Regression Model

## 🥇 Stacking Regressor

| Metric | Score |
|---|---:|
| MAE | **7.8049** |
| RMSE | **9.7166** |


# ⚔️ Bagging vs Single Decision Tree

Bagging was compared with a single Decision Tree.

| Model | Classification Accuracy |
|---|---:|
| Decision Tree | **62.98%** |
| Bagging | **72.21%** |

### 📈 Improvement

```text
Decision Tree
     ↓
62.98%

Bagging
     ↓
72.21%
```

Bagging improved the classification accuracy by approximately **9.23 percentage points**.

---

# 🗳️ Hard Voting vs Soft Voting

| Method | Accuracy |
|---|---:|
| 🥇 Hard Voting | **74.13%** |
| Soft Voting | **73.46%** |

### ✅ Result

**Hard Voting performed slightly better than Soft Voting on the test dataset.**

---

# 📊 Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 62.98% | 0.5062 | 0.5256 | 0.5157 | 0.6090 |
| Bagging | 72.21% | 0.6472 | 0.5692 | 0.6057 | 0.7739 |
| AdaBoost | 73.65% | 0.6883 | 0.5436 | 0.6074 | 0.7850 |
| Gradient Boosting | 73.85% | 0.6832 | 0.5641 | 0.6180 | 0.7896 |
| LightGBM | 72.88% | 0.6656 | 0.5564 | 0.6061 | 0.7805 |
| XGBoost | 73.56% | 0.6727 | 0.5744 | 0.6196 | 0.7875 |
| Hard Voting | 74.13% | 0.6921 | 0.5590 | 0.6184 | — |
| Soft Voting | 73.46% | 0.6770 | 0.5590 | 0.6124 | 0.7891 |
| 🏆 **Stacking** | **74.42%** | **0.6867** | **0.5846** | **0.6316** | **0.7940** |

---

# 📈 Regression Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| AdaBoost | 8.5452 | 10.5174 | 0.4080 |
| Gradient Boosting | 7.9645 | 9.9190 | 0.4734 |
| LightGBM | 7.9309 | 9.8756 | 0.4780 |
| XGBoost | 7.9205 | 9.8749 | 0.4781 |
| 🏆 **Stacking** | **7.8049** | **9.7166** | **0.4947** |

---

# 🔄 Machine Learning Workflow

```text
                    DATASET
                       ↓
              Data Understanding
                       ↓
            Feature & Target Selection
                       ↓
                Train-Test Split
                       ↓
               Data Preprocessing
                       ↓
              Numerical + Categorical
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
   CLASSIFICATION               REGRESSION
          ↓                         ↓
      Bagging                   Bagging
      AdaBoost                  AdaBoost
      Gradient Boosting         Gradient Boosting
      LightGBM                  LightGBM
      XGBoost                   XGBoost
      Voting                    Stacking
      Stacking
          ↓                         ↓
 Classification Metrics       Regression Metrics
          ↓                         ↓
          └────────────┬────────────┘
                       ↓
                Model Comparison
                       ↓
               Best Model Selection
```
| R² | **0.4947** |

### ✅ Result

**Stacking Regressor achieved the lowest prediction error and highest R² among the evaluated regression models.**

# 📌 Q6–Q32 Topics Covered

| Question | Topic |
|---|---|
| Q6 | Feature and Target Selection |
| Q7 | Train-Test Split |
| Q8 | Data Preprocessing |
| Q9 | Bagging Classifier |
| Q10 | Bagging Regressor |
| Q11 | Bagging vs Single Base Model |
| Q12 | AdaBoost Classifier |
| Q13 | AdaBoost Regressor |
| Q14 | Weak Learner Improvement |
| Q15 | Gradient Boosting Classifier |
| Q16 | Gradient Boosting Regressor |
| Q17 | Learning Rate and Estimators |
| Q18 | LightGBM Classifier |
| Q19 | LightGBM Regressor |
| Q20 | Performance and Training Efficiency |
| Q21 | XGBoost Classifier |
| Q22 | XGBoost Regressor |
| Q23 | Ensemble Performance Comparison |
| Q24 | Voting Classifier |
| Q25 | Hard Voting vs Soft Voting |
| Q26 | Stacking Classifier |
| Q27 | Stacking Regressor |
| Q28 | Classification Evaluation |
| Q29 | Regression Evaluation |
| Q30 | Ensemble Model Comparison |
| Q31 | Final Analysis and Reporting |
| Q32 | Final Evaluation and Conclusion |

---

# 📁 Project Structure

```text
SMART-OUTCOME-PREDICTOR/
│
├── README.md
│
├── project5.ipynb
│
├── dataset.5.csv
│
├── Part_A_Ensemble_Learning_Answers pro=5.docx
│
└── assets/
    │
    ├── classification_model_comparison.png
    ├── regression_r2_comparison.png
    ├── regression_rmse_comparison.png
    ├── workflow.png
    ├── notebook_output_1.png
    ├── notebook_output_2.png
    │
    └── images/
```

---

# 🖼️ Project Screenshots


## 🎯 Plot accuracy improvement

<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/e451b9fb-2c41-4e40-8545-ee1f4fac140b" />


##📈 Classification Graph
<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/0617a553-46cf-417f-b330-d92a645803e3" />


## 📊 Classification Models Comparison
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/a3c06160-23fc-4b65-8e92-5b18fdc11deb" />


## 📊 Regression Models Comparison

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/75e47d74-6956-41fd-b7d7-2a864cfc6c0e" />



## 📊Classification Models - Accuracy Comparison

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/0ab5fe9b-2b39-433c-887d-34992b0246f0" />



## 📊 Classification Models - F1-Score Comparison
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/90d2138f-e124-49c2-a8d3-0a8e145f0d33" />



## 📊 Classification Models - ROC-AUC Comparison

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/532f1788-1c20-4004-a62a-54c65e855cef" />

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🐼 Pandas | Data Analysis |
| 🔢 NumPy | Numerical Computing |
| 📊 Matplotlib | Data Visualization |
| 🤖 Scikit-learn | Machine Learning |
| ⚡ LightGBM | Gradient Boosting |
| 🚀 XGBoost | Gradient Boosting |
| 📓 Jupyter Notebook | Development |

---

# ⚙️ Installation

## 1. Install Python

Make sure Python is installed on your computer.

## 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm xgboost jupyter
```

## 3. Start Jupyter Notebook

```bash
jupyter notebook
```

## 4. Open the Project

Open:

```text
project5.ipynb
```

Make sure the dataset is available in the project folder:

```text
dataset.5.csv
```

---

# ▶️ How to Run

```text
1. Load the dataset
        ↓
2. Explore the dataset
        ↓
3. Select features and targets
        ↓
4. Split the dataset
        ↓
5. Preprocess the data
        ↓
6. Train ensemble models
        ↓
7. Generate predictions
        ↓
8. Evaluate models
        ↓
9. Compare results
        ↓
10. Select the best model
```

# 💡 Key Learnings

### 📦 Bagging

Bagging reduces variance by combining multiple independently trained models.

### 🚀 Boosting

Boosting trains models sequentially and focuses on reducing previous errors.

### ⚡ LightGBM

LightGBM provides an efficient implementation of gradient boosting.

### 🚀 XGBoost

XGBoost is a powerful and regularized gradient boosting algorithm.

### 🗳️ Voting

Voting combines predictions from multiple classification models.

### 🧩 Stacking

Stacking combines base models using a meta-model.

### 🏆 Ensemble Learning

Ensemble methods can improve prediction performance and stability compared with individual models.

---

# 🔥 Key Project Insights

## 💡 Insight 1 — Bagging Improvement

```text
Decision Tree Accuracy = 62.98%

Bagging Accuracy = 72.21%
```

Bagging provided a significant improvement over the single Decision Tree.

---

## 💡 Insight 2 — Best Classification Model

```text
Model = Stacking Classifier

Accuracy = 74.42%
F1 Score = 0.6316
ROC-AUC = 0.7940
```

---

## 💡 Insight 3 — Best Regression Model

```text
Model = Stacking Regressor

MAE = 7.8049
RMSE = 9.7166
R² = 0.4947
```

---

## 💡 Insight 4 — Voting Comparison

```text
Hard Voting = 74.13%

Soft Voting = 73.46%
```

Hard Voting performed slightly better on the test data.

# 🚀 Future Improvements

- 🔹 Hyperparameter Optimization
- 🔹 Cross-Validation
- 🔹 Feature Importance Analysis
- 🔹 SHAP Explainability
- 🔹 Model Saving using Joblib
- 🔹 Streamlit Web Application
- 🔹 REST API
- 🔹 Cloud Deployment

---

# 🏁 Conclusion

The **Smart Outcome Predictor** project demonstrates the practical application of Ensemble Learning for both classification and regression problems.

The project covers:

```text
Bagging
AdaBoost
Gradient Boosting
LightGBM
XGBoost
---
## 🏆 Recommended Models

| Task | Recommended Model |
|---|---|
| 🎯 Classification | **Stacking Classifier** |
| 📈 Regression | **Stacking Regressor** |

---

# 👨‍💻 Author

## Rahul Zala

**Python • Machine Learning • Data Science**

---

## ⭐ Support

If you found this project useful, please give the repository a **Star ⭐**.

---

<p align="center">

**Made with ❤️ by Rahul Zala**

</p>
