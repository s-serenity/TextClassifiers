import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)

    # Remove mentions and hashtags
    tweet = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', tweet)

    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)

    # Remove 'RT' (Retweet) indicator
    tweet = re.sub(r'\bRT\b', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Remove stopwords
    #     stop_words = set(stopwords.words('english'))
    #     tweet_tokens = nltk.word_tokenize(tweet)
    #     tweet = ' '.join([word for word in tweet_tokens if word not in stop_words])

    # Lemmatization
    doc = nlp(tweet)
    # Lemmatize each token and join them back into a string
    tweet = ' '.join([token.lemma_ for token in doc])

    return tweet