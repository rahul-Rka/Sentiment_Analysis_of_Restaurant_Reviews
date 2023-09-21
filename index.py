import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import accuracy_score

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load your TSV dataset (set.tsv) with tab-separated values
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to perform sentiment analysis on a text
def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to each review in the dataset
df['Sentiment'] = df['Review'].apply(analyze_sentiment)

# Calculate accuracy by comparing predicted sentiments to the "Liked" column
accuracy = accuracy_score(df['Liked'], [1 if sentiment == 'Positive' else 0 for sentiment in df['Sentiment']])

# Print the results and accuracy
print("Sentiment Analysis Results:")
print(df[['Review', 'Sentiment']])
print("\nAccuracy:", accuracy)
