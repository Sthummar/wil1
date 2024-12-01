import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit app
def main():
    st.title("Text Sentiment Analysis and Preprocessing")
    st.write("Analyze sentiment and preprocess text interactively!")

    # Input Section
    st.subheader("Enter your text")
    user_input = st.text_area("Text to analyze", placeholder="Type your text here...")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Sentiment Analysis
            sentiment = sia.polarity_scores(user_input)
            st.subheader("Sentiment Analysis Results")
            st.json(sentiment)
        else:
            st.warning("Please enter some text!")

    # Preprocessing Section
    st.subheader("Text Preprocessing")
    if user_input.strip():
        # Tokenization
        tokens = word_tokenize(user_input)
        st.write("Tokenized Text:")
        st.write(tokens)

        # Stopword Removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        st.write("Filtered Text (Stopwords Removed):")
        st.write(filtered_tokens)
    else:
        st.warning("Enter text above to preprocess!")

if __name__ == "__main__":
    main()
