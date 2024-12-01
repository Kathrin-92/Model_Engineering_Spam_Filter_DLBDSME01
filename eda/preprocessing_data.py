# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import os
import re
import string

# third-party library imports
import pandas as pd
import nltk

# download NLTK data for tokenization & for English stopwords
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# create a set of English stopwords using NLTK
english_stopwords = set(nltk.corpus.stopwords.words('english'))


# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING DATA
# ----------------------------------------------------------------------------------------------------------------------

# read data file
data = pd.read_csv('data/raw/SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

# drop duplicate entries with the same message
data.drop_duplicates(subset='message', keep='last', inplace=True)


# define pre-cleaning function for 'message' column
def clean_text(text):
    text = text.replace(r"\n", "") # remove line breaks
    text = re.sub(r"\s{2,}", " ", text) # remove double white space
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special characters and punctuation
    text = re.sub(r"[\d-]", "", text)  # remove numbers
    text = text.lower()  # convert to lowercase
    text = nltk.word_tokenize(text) # tokenize
    text = [token for token in text if token not in english_stopwords] # remove stopwords
    return text


# apply cleaning function to the 'message' column
data['message_cleaned'] = data['message'].apply(lambda x: clean_text(x))

# convert list of words to a single string
data['message_cleaned'] = data['message_cleaned'].apply(lambda x: ' '.join(x))

# add several calculated columns based on EDA results
data['char_count'] = data['message'].apply(lambda x: sum(1 for char in x if char in string.ascii_letters))
data['char_count_cleansed'] = data['message_cleaned'].apply(lambda x: sum(1 for char in x if char in string.ascii_letters))
data['special_char_count'] = data['message'].apply(lambda x: sum(1 for char in x if char in string.punctuation))
data['special_char_ratio'] = data['special_char_count'] / data['char_count_cleansed']
data['word_count_cleansed'] = data['message_cleaned'].apply(lambda x: len(x.split()))
data['word_count_ratio'] = data['word_count_cleansed'] / data['char_count_cleansed']
data.insert(0, 'label_no', data['label'].apply(lambda x: 1 if x == 'spam' else 0)) # create columns with number for label

# save cleaned data in a csv file
if not os.path.exists(os.path.join('data/processed', 'data_cleaned.csv')):
    data.to_csv(os.path.join('data/processed', 'data_cleaned.csv'), index=False)
