# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# third-party library imports
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ----------------------------------------------------------------------------------------------------------------------
# LOADING DATA
# ----------------------------------------------------------------------------------------------------------------------

# read data file
data = pd.read_csv('data/raw/SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

# checking data structure
data.info() # --> RangeIndex:5572 entries, 0 to 5571
print(data.shape) # --> (5571, 2)
print(data.columns)
data.head(5)

# check for missing data using isnull
data.isnull().sum() # --> none found


# ----------------------------------------------------------------------------------------------------------------------
# DISTRIBUTION OF LABELS
# ----------------------------------------------------------------------------------------------------------------------

label_distribution = data.groupby('label').size()
print(f'Label Distribution of Ham vs Spam: {label_distribution}') # --> ham: 4825 (86,6%) // spam: 747 (13,4%)

# look for duplicate content
duplicates = data[data.duplicated(subset='message', keep=False)]
duplicates = duplicates.sort_values(by='message')
duplicate_counts = duplicates.groupby('message').size() # --> 281 messages are duplicates; 402 rows would be deleted if dropped
duplicate_counts_label = duplicates.groupby('label').size() # --> 503 ham, 181 spam
print(f'Messages with count of duplicates: {duplicate_counts}')
print(f'Number of duplicates total per label: {duplicate_counts_label}')

# most duplicate messages are labeled as 'ham' suggesting they are likely artifacts from a non-clean dataset
# they are therefore treated as noise and removed --> also applied in preprocessing_data.py
data.drop_duplicates(subset='message', keep='last', inplace=True) # --> (5169, 2)
data.groupby('label').describe()
label_distribution_without_dups = data.groupby('label').size()
print(f'Label Distribution of Ham vs Spam without Duplicates: {label_distribution_without_dups}') # --> ham: 4516 (87,4%) // spam: 653 (12,6%%)

# visualization of label distribution
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
# other
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ----------------------------------------------------------------------------------------------------------------------
# TEXT LENGTH OF HAM VS SPAM
# use char_count and char_count_cleansed columns from preprocessing
# ----------------------------------------------------------------------------------------------------------------------

# read preprocessed data file
data_df = pd.read_csv('data/processed/data_cleaned.csv', sep=',')

# average message length: ham vs spam
# ham: 53.346988 characters // spam: 93.364472 characters
avg_char_per_label = data_df.groupby('label')['char_count'].mean()
print(f'Average message length (number of characters) per label: {avg_char_per_label}')

# plot the 'char_count' column (raw character count)
plt.figure(figsize=(13, 6))
plt.subplot(1, 2, 1)
plt.hist(data_df[data_df['label'] == 'ham']['char_count'], bins=25, alpha=0.5, label='ham', color='mediumseagreen', density=True)
plt.hist(data_df[data_df['label'] == 'spam']['char_count'], bins=25, alpha=0.5, label='spam', color='tomato', density=True)
# styling
plt.legend()
plt.ylabel(ylabel='frequency', fontsize=10, weight='bold')
plt.xlabel(xlabel='character count', fontsize=10, weight='bold')
plt.title('Character Count Distribution (normalized) - Raw', fontsize=12, weight='bold')

# plot the 'char_count_cleansed' column (cleaned character count)
plt.subplot(1, 2, 2)
plt.hist(data_df[data_df['label'] == 'ham']['char_count_cleansed'], bins=25, alpha=0.5, label='ham', color='mediumseagreen', density=True)
plt.hist(data_df[data_df['label'] == 'spam']['char_count_cleansed'], bins=25, alpha=0.5, label='spam', color='tomato', density=True)
# styling
plt.legend()
plt.ylabel(ylabel='frequency', fontsize=10, weight='bold')
plt.xlabel(xlabel='character count', fontsize=10, weight='bold')
plt.title('Character Count Distribution (normalized) - Cleaned', fontsize=12, weight='bold')
plt.tight_layout()


# ----------------------------------------------------------------------------------------------------------------------
# SPECIAL CHARACTERS
# ----------------------------------------------------------------------------------------------------------------------

# count special characters and numbers used by label (ham vs spam)
# ham: 17297 special char // spam: 3685 special char
special_char_sum_by_label = data_df.groupby('label')['special_char_count'].sum()
print(f'Number of special characters per label: {special_char_sum_by_label}')

# as there are more ham messages in the dataset, calculate average because absolute numbers are not meaningful
# --> ham: 3.830159 // spam: 5.643185 --> spam messages seem to use more special characters on average
avg_special_char_per_label = data_df.groupby('label')['special_char_count'].mean()
print(f'Average Number of special characters per label: {avg_special_char_per_label}')

# calculate the ratio of special characters for each message
# histograms of the text lengths lead to the conclusion that the cleaning of the messages leads to the spam texts
# being much shorter than before; therefore the ratio is calculated with the original text length
# --> column also included in preprocessing
data_df['special_char_ratio'] = data_df['special_char_count'] / data_df['char_count']

# scatter plot to show the relationship between special character and text length
fig, ax = plt.subplots(figsize=(8, 6))

plt.scatter(
    data_df[data_df['label'] == 'ham']['char_count'],
    data_df[data_df['label'] == 'ham']['special_char_count'],
    alpha=0.5,
    label='ham',
    color='mediumseagreen')

plt.scatter(
    data_df[data_df['label'] == 'spam']['char_count'],
    data_df[data_df['label'] == 'spam']['special_char_count'],
    alpha=0.5,
    label='spam',
    color='tomato')

# styling
# y-axis
ax.set_ylabel('special character count', fontsize=10, weight='bold')
# x-axis
ax.set_xlabel('total character count', fontsize=10, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation='horizontal')
# title
ax.set_title('Special Characters vs Text Length', fontsize=14, weight='bold')
# other
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# --> the scatterplot however seems to indicate that there is not really a heavy relation between the usage of special
# characters and text length between spam vs ham; the text length itself seems to be a bigger factor


# ----------------------------------------------------------------------------------------------------------------------
# WORD COUNTS
# ----------------------------------------------------------------------------------------------------------------------

# ham messages
ham_text = data_df[data_df['label'] == 'ham']['message_cleaned'].str.cat(sep=" ")
ham_words = ham_text.split()
ham_word_series = pd.Series(ham_words)
# - average number of words
avg_word_count_ham = len(ham_words) / data_df[data_df['label'] == 'ham'].shape[0]
print(f'Avg. number of words in Ham messages: {avg_word_count_ham}') # 8.051

# spam messages - most frequent words
spam_text = data_df[data_df['label'] == 'spam']['message_cleaned'].str.cat(sep=" ")
spam_words = spam_text.split()
spam_word_series = pd.Series(spam_words)
# - average number of words
avg_word_count_spam = len(spam_words) / data_df[data_df['label'] == 'spam'].shape[0]
print(f'Avg. number of words in Spam messages: {avg_word_count_spam}') # 14.152
# --> add word count as feature to data in preprocessing

# get word counts from preprocessed data and display as histogram
plt.figure(figsize=(10, 4))
plt.hist(data_df[data_df['label'] == 'ham']['word_count_cleansed'], bins=20, alpha=0.8, label='ham', color='mediumseagreen', density=True)
plt.hist(data_df[data_df['label'] == 'spam']['word_count_cleansed'], bins=20, alpha=0.8, label='spam', color='tomato', density=True)
# styling
plt.legend()
plt.ylabel(ylabel='frequency', fontsize=10, weight='bold')
plt.xlabel(xlabel='word count', fontsize=10, weight='bold')
plt.title('Word Count Distribution (normalized)', fontsize=12, weight='bold')


# ----------------------------------------------------------------------------------------------------------------------
# TOP WORD COUNTS & WORD CLOUD
# ----------------------------------------------------------------------------------------------------------------------

# ham - most frequent words
top_15_words_ham = ham_word_series.value_counts().head(15)
top_15_words_ham = top_15_words_ham.sort_values(ascending=True)

# spam - most frequent words
top_15_words_spam = spam_word_series.value_counts().head(15)
top_15_words_spam = top_15_words_spam.sort_values(ascending=True)

## horizontal bar plot
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax1 = ax[0]
ax2 = ax[1]

# ham
top_15_words_ham.plot(kind='barh', ax=ax1, color='navy')
ax1.set_title(f'Top 15 Most Used Words in Ham Messages', fontsize=10, weight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# spam
top_15_words_spam.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_title(f'Top 15 Most Used Words in Spam Messages', fontsize=10, weight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()


## word cloud
plt.figure(figsize=(10, 4))

# ham
plt.subplot(1, 2, 1)
ham_word_cloud = WordCloud(background_color='white', colormap="gist_earth").generate(data_df[data_df['label']=='ham']['message_cleaned'].str.cat(sep=" "))
plt.imshow(ham_word_cloud)
plt.title(f'Ham Messages Word Cloud', fontsize=10, weight='bold', pad=20)
plt.axis("off")

# spam
plt.subplot(1, 2, 2)
spam_word_cloud = WordCloud(background_color='white', colormap="gist_earth").generate(data_df[data_df['label']=='spam']['message_cleaned'].str.cat(sep=" "))
plt.imshow(spam_word_cloud)
plt.title(f'Spam Messages Word Cloud', fontsize=10, weight='bold', pad=20)
plt.axis("off")
plt.tight_layout()


# ----------------------------------------------------------------------------------------------------------------------
# CHECK CORRELATIONS OF NEW NUMERICAL FEATURES IN PREPROCESSED DATA
# to understand the relationships between features to avoid multicollinearity/redundancy in feature selection
# ----------------------------------------------------------------------------------------------------------------------

correlation_matrix = data_df[['char_count_cleansed', 'word_count_cleansed', 'word_count_ratio', 'special_char_count', 'special_char_ratio']].corr()

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(correlation_matrix, cmap='Oranges')
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=6)
ax.set_yticklabels(correlation_matrix.columns, fontsize=6)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="#cccccc")

ax.set_title("Correlation Matrix of Numerical Features", fontsize=10, weight='bold', pad=20)
plt.show()
