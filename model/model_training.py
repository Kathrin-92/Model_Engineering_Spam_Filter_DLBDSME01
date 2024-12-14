# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# third-party library imports
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

# ----------------------------------------------------------------------------------------------------------------------
# LOG MODEL WITH MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# in terminal, run local tracking server: mlflow server --host 127.0.0.1 --port 8080
# set the tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# create MLflow Experiment
## baseline model provides reference point for evaluating more complex models
mlflow.set_experiment("20241214_Baseline_Model_v1")

# use MLflow autologging functionality
mlflow.autolog()

with mlflow.start_run():
    mlflow.set_tag("iteration", "6")
    mlflow.set_tag("model", "Logistic Regression")
    mlflow.set_tag("experiment_phase", "baseline_model_adjusted_training")
    mlflow.set_tag("data_split", "60% train, 20% test, 20% validation")

    ## baseline model: logistic regression
    # EDA shows that text length is strongly indicative of spam messages
    # logistic regression is designed for binary classification tasks, straightforward/quick to train and
    # provides interpretable results

    # load necessary data from preprocessed file
    df = pd.read_csv('data/processed/data_cleaned.csv', sep=',')
    X = df[['char_count_cleansed']]
    y = df['label_no']

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # additionally split training data into validation and training set
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    # configure logistic regression model with early stopping; helps to prevent overfitting
    # https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
    logistic_reg_model = LogisticRegression(
        solver='liblinear',  # suitable for small datasets
        # solver='saga',
        class_weight='balanced',  # adjust for class imbalance; adjusts weight of each class inversely proportional to its frequency
        max_iter=1000,
        C=2.0,
        random_state=42
    )
    logistic_reg_model.fit(X_train, y_train)

    # make predictions and validation on validation_set
    y_validate_pred = logistic_reg_model.predict(X_validate)

    # evaluate the model
    iteration_numb = logistic_reg_model.n_iter_
    accuracy_score = accuracy_score(y_validate, y_validate_pred)
    mse = mean_squared_error(y_validate, y_validate_pred)

    # log everything
    mlflow.log_metric("Number of iterations", iteration_numb)
    mlflow.log_metric("Validation Accuracy", accuracy_score)
    mlflow.log_metric("Validation MSE", mse)

    print("Number of iterations:", iteration_numb)
    print("Validation Accuracy:", accuracy_score)
    print("Validation MSE:", mse)
    print("Validation Classification Report:\n", classification_report(y_validate, y_validate_pred))
