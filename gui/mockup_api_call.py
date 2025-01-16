# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

import requests # for test message

# uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# ----------------------------------------------------------------------------------------------------------------------
# MOCKUP CALL TO FASTAPI
# ----------------------------------------------------------------------------------------------------------------------
endpoint_url = 'http://127.0.0.1:8000/get_spam_prediction'

# Define the input data
input_data = {"message": "this is a test message"}

# Make the POST request
response = requests.post(endpoint_url, json=input_data)
print(response)



"""
# ----------------------------------------------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------------------------------------------
def mock_model_predict(input_text):
    return f"Predicted response for '{input_text}'"


# ----------------------------------------------------------------------------------------------------------------------
# CLI - MOCKUP FOR MODEL INTERACTION
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # program starts and intro messages introduces the simulation
    intro_message = \
        "\n************************************************************************************\n" \
        "Welcome to the Spam Risk Simulator" \
        "\n************************************************************************************\n" \
        "\nThis tool is designed to help simulate how customer messages are evaluated " \
        "for potential spam risk using our prediction model." \
        "\nEnter a mock message just as a customer might, and the system will return " \
        "a spam prediction, " \
        "including the likelihood of spam and its associated risk level." \
        "\nIf you’d like to exit the simulation at any time, simply type 'exit' or 'quit'.\n" \
        "\n************************************************************************************\n" \
        "Let’s begin! Type in a moc customer message to see the prediction in action." \
        "\n************************************************************************************\n"

    print(intro_message)

    while True:
        # ask the user to input a message/question
        user_input = questionary.text("Write a mock message/question:").ask()

        # check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the program. Goodbye!")
            break

        # get prediction from the model
        prediction = mock_model_predict(user_input)

        # display the prediction result
        print(f"\nPrediction: {prediction}\n")


if __name__ == "__main__":
    main()"""