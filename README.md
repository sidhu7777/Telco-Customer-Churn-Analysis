

---

# Telco Customer Churn Analysis

This repository presents an end-to-end machine learning workflow for predicting customer churn in the telecommunications sector. It covers data preprocessing, exploratory analysis, model building using XGBoost, and model interpretation using ELI5 and SHAP.

## Project Objective

The goal is to predict whether a customer will churn based on historical data and to interpret the model's decision-making process. This allows for targeted business strategies to reduce churn and improve customer retention.

## Dataset

The analysis uses the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn), containing customer demographic data, service usage information, and account details.

## Workflow Summary

### 1. Data Exploration and Preprocessing

* Checked for missing values and duplicates.
* Conducted EDA with visualizations to understand feature distributions and churn correlations.
* Created new categorical features such as `tenure_group`.
* Encoded categorical variables for model compatibility.

### 2. Modeling with XGBoost

* Used `XGBClassifier` from the `xgboost` library with `use_label_encoder=False` and `eval_metric='logloss'`.
* The model was trained on a split of the data (`X_train`, `y_train`) and evaluated on a test set.
* Performance metrics included:

  * **Accuracy**
  * **ROC-AUC Score**
  * **Classification Report** (Precision, Recall, F1-score)

Example snippet:

```python
from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 3. Model Evaluation

* Used `classification_report`, `accuracy_score`, and `roc_auc_score` from `sklearn.metrics`.
* Found strong performance on the test set, indicating the model's reliability in distinguishing churners from non-churners.

### 4. Model Explainability

#### ELI5

* Installed and used `eli5` to interpret feature importances from the XGBoost model.
* Generated HTML reports for weight-based feature contributions.
* Enabled intuitive insight into which features most influenced the churn predictions.

#### SHAP (SHapley Additive exPlanations)

* Used the `shap` library to compute SHAP values for individual predictions and global explanations.
* Plotted summary and dependence plots to visualize feature impacts.
* SHAP provided transparency at both the instance and dataset level.

Example snippet:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/telco-churn-analysis.git
   cd telco-churn-analysis
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook Final_project.ipynb
   ```

## Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* xgboost
* scikit-learn
* eli5
* shap

## License

This project is open-source and available under the MIT License.


