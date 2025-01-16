# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import joblib # for mockup api call
import uvicorn # for local run


# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE TRAINED MODEL FROM MLFLOW
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

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') # just for test/mockup
#tfidf_vectorizer = TfidfVectorizer()

# ----------------------------------------------------------------------------------------------------------------------
# START API AND DEFINE INPUT AND OUTPUT
# ----------------------------------------------------------------------------------------------------------------------

# for local run
def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="debug", reload=True)
    server = uvicorn.Server(config)
    server.run()

app = FastAPI(title="Spam Classification API")

# define model for input data
class InputData(BaseModel):
    message: str

# define model for output data
class PredictionResult(BaseModel):
    message: str
    spam_probability: float
    risk_category: str


# ----------------------------------------------------------------------------------------------------------------------
# DEFINE API ENDPOINTS
# ----------------------------------------------------------------------------------------------------------------------

# define the index route
@app.get('/')
async def index():
    return {
        "Message": "Welcome to the Spam Classification API! "
                   "This API connects a registered MLflow model to a mockup CLI simulator that returns predictions "
                   "for customer messages, including spam probability and risk levels.",
        "Instructions": "To explore the API's documentation and available endpoints, "
                        "please visit: http://localhost:8000/docs"
    }

# define the prediction route
@app.post('/get_spam_prediction', response_model=list[PredictionResult])
async def predict(input_data: InputData):
    try:
        # convert input data to numpy array / reshape / preorocess data
        message = input_data.message
        #message_transformed = tfidf_vectorizer.fit_transform([message])
        message_transformed = tfidf_vectorizer.transform([message])


        # make predictions
        spam_probability = mnb_model.predict_proba(message_transformed)[:, 1]

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
        risk_category = assign_risk_category(spam_probability)

        results = [
            PredictionResult(
                message=message,
                spam_probability=round(spam_probability[0], 2),
                risk_category=risk_category
            )
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(e)}")

# for local run
if __name__ == "__main__":
    run_server()