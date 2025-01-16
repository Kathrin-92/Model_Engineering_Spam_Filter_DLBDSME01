# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# third-party library imports
import mlflow
import pandas as pd
import joblib # needed to get same data split for test-predicitons in model_predictions.py file
from imblearn.over_sampling import RandomOverSampler

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ----------------------------------------------------------------------------------------------------------------------
# SETUP MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# in terminal, run local tracking server: mlflow server --host 127.0.0.1 --port 8080
# set the tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


# ----------------------------------------------------------------------------------------------------------------------
# BASELINE LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------------------------------------------------------------------

# create MLflow Experiment
## baseline model provides reference point for evaluating more complex models
mlflow.set_experiment("20241214_Baseline_Model_v1")

# use MLflow autologging functionality
mlflow.autolog()

with mlflow.start_run():
    mlflow.set_tag("iteration", "7")
    mlflow.set_tag("model", "Logistic Regression")
    mlflow.set_tag("experiment_phase", "baseline_model_with_resampling")
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
    joblib.dump(X_test, 'X_test_baseline.pkl')
    joblib.dump(y_test, 'y_test_baseline.pkl')

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
    precision = precision_score(y_validate, y_validate_pred, pos_label=1)
    recall = recall_score(y_validate, y_validate_pred, pos_label=1)

    # log everything
    mlflow.log_metric("Number of iterations", iteration_numb)
    mlflow.log_metric("Validation Accuracy", accuracy_score)
    mlflow.log_metric("Validation MSE", mse)
    mlflow.log_metric("Validation Precision", precision)
    mlflow.log_metric("Validation Recall", recall)

    print("Number of iterations:", iteration_numb)
    print("Validation Accuracy:", accuracy_score)
    print("Validation MSE:", mse)
    print("Validation Precision:", precision)
    print("Validation Recall:", recall)
    print("Validation Classification Report:\n", classification_report(y_validate, y_validate_pred))


# ----------------------------------------------------------------------------------------------------------------------
# ADVANCED MODEL = MULTINOMIAL NAIVE BAYES MODEL
# ----------------------------------------------------------------------------------------------------------------------

# create MLflow Experiment
mlflow.set_experiment("20241221_MNB_Model_v1")

# use MLflow autologging functionality
mlflow.autolog()

with ((mlflow.start_run())):
    mlflow.set_tag("iteration", "3")
    mlflow.set_tag("model", "Multinomial Naive Bayes")
    mlflow.set_tag("experiment_phase", "advanced_model_adjusted_training")
    mlflow.set_tag("data_split", "60% train, 20% test, 20% validation")

    ## advanced model: Multinomial Naive Bayes
    # variant of the Naive Bayes classifier that is commonly used for text classification tasks
    # works well with bag-of-words or TF-IDF representations of text
    # predicts by analyzing frequency of words in message
    # analyzes which specific words are more likely to appear in spam vs ham messages, making it more sensitive
    # to the content of the text rather than just its length

    # load necessary data from preprocessed file and vectorize the message with tfidf
    df = pd.read_csv('data/processed/data_cleaned.csv', sep=',')
    df = df.dropna(subset=['message_cleaned'])

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['message_cleaned'])
    y = df['label_no']

    # split data into training and testing sets
    X_train_mnb, X_test_mnb, y_train_mnb, y_test_mnb = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # additionally split training data into validation and training set
    X_train_mnb, X_validate_mnb, y_train_mnb, y_validate_mnb = train_test_split(X_train_mnb, y_train_mnb, test_size=0.25, random_state=42, stratify=y_train_mnb)
    joblib.dump(X_test_mnb, 'X_test_mnb.pkl')
    joblib.dump(y_test_mnb, 'y_test_mnb.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    # resample only the training data
    resampler = RandomOverSampler(random_state=42)
    X_train_mnb_resampled, y_train_mnb_resampled = resampler.fit_resample(X_train_mnb, y_train_mnb)
    # split the resampled data into validation and training sets
    # validation set should not be resampled to avoid introducing any bias into the validation or test data
    X_train_mnb_resampled_final, X_validate_mnb, y_train_mnb_resampled_final, y_validate_mnb = train_test_split(
    X_train_mnb_resampled, y_train_mnb_resampled, test_size=0.25, random_state=42, stratify=y_train_mnb_resampled)

    # configure Multinomial Naive Bayes
    mnb_model = MultinomialNB(
        alpha=1.0,
        # class_prior=[0.8, 0.2],
        fit_prior=True
    )
    # mnb_model.fit(X_train_mnb, y_train_mnb)
    mnb_model.fit(X_train_mnb_resampled_final, y_train_mnb_resampled_final)

    # make predictions and validation on validation_set
    y_validate_pred_mnb = mnb_model.predict(X_validate_mnb)

    # evaluate the model
    accuracy_score = accuracy_score(y_validate_mnb, y_validate_pred_mnb)
    mse = mean_squared_error(y_validate_mnb, y_validate_pred_mnb)
    precision = precision_score(y_validate_mnb, y_validate_pred_mnb, pos_label=1)
    recall = recall_score(y_validate_mnb, y_validate_pred_mnb, pos_label=1)

    # log everything
    mlflow.log_metric("Validation Accuracy", accuracy_score)
    mlflow.log_metric("Validation MSE", mse)
    mlflow.log_metric("Validation Precision", precision)
    mlflow.log_metric("Validation Recall", recall)

    print("Validation Accuracy:", accuracy_score)
    print("Validation MSE:", mse)
    print("Validation Precision:", precision)
    print("Validation Recall:", recall)
    print("Validation Classification Report:\n", classification_report(y_validate_mnb, y_validate_pred_mnb))
