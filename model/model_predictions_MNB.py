# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# third-party library imports
import joblib
import pandas as pd
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, accuracy_score, mean_squared_error, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)


# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE TRAINED ADVANCED MODEL FROM MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# load the model
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name = "Multinomial Naive Bayes - Spam Detection"
model_version = 1
mnb_model = None

try:
    model_uri = f"models:/{model_name}/{model_version}"
    mnb_model =  mlflow.sklearn.load_model(model_uri)
    print("Model successfully loaded!")
except MlflowException as e:
    print(f"Failed to load model: {str(e)}")

mlflow.set_experiment("20241221_Advanced_Model_v1_Test-Predictions")

with mlflow.start_run():
    mlflow.set_tag("model", "Multinomial Naive Bayes")
    mlflow.set_tag("experiment_phase", "advanced_model_test_predictions")

# ----------------------------------------------------------------------------------------------------------------------
# PREDICTIONS WITH ADVANCED MODEL
# ----------------------------------------------------------------------------------------------------------------------

    df = pd.read_csv('data/processed/data_cleaned.csv', sep=',')
    X_test_mnb = joblib.load('X_test_mnb.pkl')
    y_test_mnb = joblib.load('y_test_mnb.pkl')

    # make prediction with test data on baseline model
    y_test_pred_mnb = mnb_model.predict(X_test_mnb)
    spam_probabilities = mnb_model.predict_proba(X_test_mnb)[:, 1]

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
    test_results = pd.DataFrame(X_test_mnb)
    test_results["true_label"] = y_test_mnb.values
    test_results["spam_probability"] = spam_probabilities
    test_results["risk_category"] = risk_categories
    print(test_results)

    accuracy_score = accuracy_score(y_test_mnb, y_test_pred_mnb)
    mse = mean_squared_error(y_test_mnb, y_test_pred_mnb)
    precision = precision_score(y_test_mnb, y_test_pred_mnb, pos_label=1)
    recall = recall_score(y_test_mnb, y_test_pred_mnb, pos_label=1)

    # log everything
    mlflow.log_metric("Test Accuracy", accuracy_score)
    mlflow.log_metric("Test MSE", mse)
    mlflow.log_metric("Test Precision", precision)
    mlflow.log_metric("Test Recall", recall)
    print("Validation Classification Report:\n", classification_report(y_test_mnb, y_test_pred_mnb))

# ----------------------------------------------------------------------------------------------------------------------
# VISUALIZATIONS WITH ADVANCED MODEL
# ----------------------------------------------------------------------------------------------------------------------

## (1) CONFUSION MATRIX
# compute confusion matrix
cm = confusion_matrix(y_test_mnb, y_test_pred_mnb, normalize='all')

# plot as heatmap
labels = ['ham', 'spam']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, linewidth=.5)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Normalized Confusion Matrix (MNB)', fontsize=10, weight='bold', pad=20)


## (2) PROPORTION IN RISK CATEGORIES
# display proportion of messages in each risk_category for each true_label_name (ham / spam)
# count the number of messages for each combination; normalize counts
pivot_table = pd.crosstab(test_results['true_label'], test_results['risk_category'])
custom_order = ['high-risk', 'moderate-high risk', 'medium risk', 'moderate-low risk', 'low-risk']
pivot_table = pivot_table[custom_order]
proportions = pivot_table.div(pivot_table.sum(axis=1), axis=0)
print(proportions)
print(pivot_table)

ax = proportions.plot(kind='bar', stacked=True, figsize=(8, 6), colormap="Oranges_r")
for container in ax.containers:
    ax.bar_label(container, labels=[f'{v:.2f}' for v in container.datavalues], label_type='center', color='grey')
plt.title("Proportion of Risk Levels by True Labels (MNB)", fontsize=10, weight='bold')
plt.xlabel("True Label")
plt.ylabel("Proportion")
plt.legend(title="Risk Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
