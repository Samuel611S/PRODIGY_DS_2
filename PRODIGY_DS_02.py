import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the CSV file without headers
data = pd.read_csv('twitter_training.csv', header=None)

# Define and add column titles
column_titles = ['ID', 'Game', 'Sentiment', 'Text']
data.columns = column_titles

# Convert the text column to string
data['Text'] = data['Text'].astype(str)

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply sentiment analysis
def get_sentiment_score(text):
    return sid.polarity_scores(text)['compound']

# Get sentiment scores and classify sentiments
data['sentiment_score'] = data['Text'].apply(get_sentiment_score)
data['sentiment'] = data['sentiment_score'].apply(lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'))

# Display the first few rows with the sentiment scores
print(data.head())
print(data.columns)

# Plot the distribution of sentiment
sns.countplot(x='sentiment', data=data)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
