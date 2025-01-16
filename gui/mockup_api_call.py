# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

import requests # for testing the api endpoint with a test message


# ----------------------------------------------------------------------------------------------------------------------
# MOCKUP CALL TO FASTAPI
# ----------------------------------------------------------------------------------------------------------------------
endpoint_url = 'http://127.0.0.1:8000/get_spam_prediction'

# Define the input data
input_data = {"message": "this is a test message"}

# Make the POST request
response = requests.post(endpoint_url, json=input_data)
print(response.json())
