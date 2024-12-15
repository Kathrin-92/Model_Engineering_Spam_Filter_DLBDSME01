# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# third-party library imports
import joblib
import pandas as pd
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, precision_score, recall_score


# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE TRAINED BASELINE LOGISTIC REGRESSION MODEL FROM MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# load the model
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name = "Baseline Logistic Regression - Spam Detection"
model_version = 1
logistic_reg_model = None

try:
    model_uri = f"models:/{model_name}/{model_version}"
    logistic_reg_model =  mlflow.sklearn.load_model(model_uri)
    print("Model successfully loaded!")

except MlflowException as e:
    print(f"Failed to load model: {str(e)}")

mlflow.set_experiment("20241215_Baseline_Model_v1_Test-Predictions")

# use MLflow autologging functionality
mlflow.autolog()

with mlflow.start_run():
    mlflow.set_tag("model", "Logistic Regression")
    mlflow.set_tag("experiment_phase", "baseline_model_test_predictions")

# ----------------------------------------------------------------------------------------------------------------------
# PREDICTIONS WITH BASELINE LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------------------------------------------------------------------

    df = pd.read_csv('data/processed/data_cleaned.csv', sep=',')
    X_test_baseline = joblib.load('X_test_baseline.pkl')
    y_test_baseline = joblib.load('y_test_baseline.pkl')

    # make prediction with test data on baseline model
    y_test_pred = logistic_reg_model.predict(X_test_baseline)
    spam_probabilities = logistic_reg_model.predict_proba(X_test_baseline)[:, 1]

    # thresholds for risk categories
    def assign_risk_category(prob):
        if prob > 0.9:
            return "high-risk"
        elif prob > 0.75:
            return "moderate-high risk"
        elif prob > 0.5:
            return "medium risk"
        elif prob > 0.25:
            return "moderate-low risk"
        else:
            return "low-risk"

    # assign risk categories
    risk_categories = [assign_risk_category(prob) for prob in spam_probabilities]

    # put everything together as a dataframe
    test_results = pd.DataFrame(X_test_baseline)
    test_results["message"] = df.loc[X_test_baseline.index, "message"].values
    test_results["true_label"] = y_test_baseline.values
    test_results["true_label_name"] = df.loc[X_test_baseline.index, "label"].values
    test_results["spam_probability"] = spam_probabilities
    test_results["risk_category"] = risk_categories
    # print(test_results)

    accuracy_score = accuracy_score(y_test_baseline, y_test_pred)
    mse = mean_squared_error(y_test_baseline, y_test_pred)
    precision = precision_score(y_test_baseline, y_test_pred, pos_label=1)
    recall = recall_score(y_test_baseline, y_test_pred, pos_label=1)

    # log everything
    mlflow.log_metric("Test Accuracy", accuracy_score)
    mlflow.log_metric("Test MSE", mse)
    mlflow.log_metric("Test Precision", precision)
    mlflow.log_metric("Test Recall", recall)
    print("Validation Classification Report:\n", classification_report(y_test_baseline, y_test_pred))


# ----------------------------------------------------------------------------------------------------------------------
# VISUALIZATIONS WITH BASELINE LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------------------------------------------------------------------

# display proportion of messages in each risk_category for each true_label_name (ham / spam)
# count the number of messages for each combination; normalize counts
pivot_table = pd.crosstab(test_results['true_label_name'], test_results['risk_category'])
custom_order = ['high-risk', 'moderate-high risk', 'medium risk', 'moderate-low risk', 'low-risk']
pivot_table = pivot_table[custom_order]
proportions = pivot_table.div(pivot_table.sum(axis=1), axis=0)
print(proportions)
print(pivot_table)

ax = proportions.plot(kind='bar', stacked=True, figsize=(8, 6), colormap="Oranges_r")
for container in ax.containers:
    ax.bar_label(container, labels=[f'{v:.2f}%' for v in container.datavalues], label_type='center', color='grey')
plt.title("Proportion of Risk Levels by True Labels", fontsize=10, weight='bold')
plt.xlabel("True Label")
plt.ylabel("Proportion")
plt.legend(title="Risk Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE TRAINED ADVANCED MODEL FROM MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# PREDICTIONS WITH ADVANCED MODEL
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# VISUALIZATIONS WITH ADVANCED MODEL
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# COMPARING BASELINE WITH ADVANCED MODEL
# ----------------------------------------------------------------------------------------------------------------------
