# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# LOADING DATA
# ----------------------------------------------------------------------------------------------------------------------

# read data file
data = pd.read_csv('data/raw/SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

# checking data structure
# data.info() # --> RangeIndex: 5571 entries, 0 to 5570
# print(data.shape) # --> (5571, 2)
# print(data.columns)
# data.head(5)

# check for missing data using isnull
print(data.isnull().sum()) # --> none found


# ----------------------------------------------------------------------------------------------------------------------
# DISTRIBUTION OF LABELS & REMOVE DUPLICATES
# ----------------------------------------------------------------------------------------------------------------------

label_distribution = data.groupby('label').size()
# print(label_distribution) # --> ham: 4825 (86,6%) // spam: 747 (13,4%)

# look for duplicate content
duplicates = data[data.duplicated(subset='message', keep=False)]
duplicates = duplicates.sort_values(by='message')
duplicate_counts = duplicates.groupby('message').size() # --> 281 messages are duplicates; 402 rows would be deleted if dropped
duplicate_counts_label = duplicates.groupby('label').size() # --> 503 ham, 181 spam
# print(duplicate_counts)
# print(duplicate_counts_label)

# most duplicate messages are labeled as "ham" suggesting they are likely artifacts from a non-clean dataset
# they are therefore treated as noise and removed
# drop duplicate entries with the same message
data.drop_duplicates(subset='message', keep='last', inplace=True) # --> (5169, 2)
data.groupby('label').describe()
label_distribution_without_dups = data.groupby('label').size()
# print(label_distribution_without_dups) # --> ham: 4516 (87,4%) // spam: 653 (12,6%%)

# visualisation of label distribution
label_counts = data['label'].value_counts()
label_percentages = data['label'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(8, 6))
label_counts.plot(kind='bar', color=['mediumseagreen', 'tomato'], ax=ax)

# add percentage labels on top of the bars
for i, count in enumerate(label_counts):
    ax.text(i, count-150, f'{label_percentages.iloc[i]:.2f}%', ha='center', va='center_baseline', fontsize=12)

# styling
# y-axis
ax.set_ylabel('count', fontsize=10, weight='bold', loc='top', rotation='horizontal', labelpad = -20)
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks()])

# x-axis
ax.set(xlabel='')
ax.set_xticklabels(ax.get_xticklabels(), rotation='horizontal')

# title
ax.set_title('Distribution of Spam and Ham Labels', fontsize=14, weight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# WORD COUNTS & ROUGH CLEANING
# ----------------------------------------------------------------------------------------------------------------------

# Wortlänge
#word_length_spam
#word_length_ham

# Wortanzahl pro Message
#word_counts_spam
#word_counts_ham

# Die am häufigsten auftretendsten Wörter



# ----------------------------------------------------------------------------------------------------------------------
# WORD CLOUD
# ----------------------------------------------------------------------------------------------------------------------
