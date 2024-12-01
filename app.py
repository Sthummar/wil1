import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

# Run the download function to ensure resources are available
download_nltk_resources()

# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # User input
    user_input = st.text_area("Enter text to analyze sentiment:")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(user_input)

            # Extract negative, neutral, and positive sentiment scores
            negative_score = sentiment['neg']
            neutral_score = sentiment['neu']
            positive_score = sentiment['pos']

            # Display the results
            st.write("Negative Sentiment:", negative_score)
            st.write("Neutral Sentiment:", neutral_score)
            st.write("Positive Sentiment:", positive_score)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
