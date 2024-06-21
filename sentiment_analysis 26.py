Task-26 Sentimental Analysis
# Loading Libraries
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report, ConfusionMatrixDisplay
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob
import re
nlp = spacy.load("en_core_web_sm") 
nlp.add_pipe('spacytextblob')
# Loading dataset
df = pd.read_csv("amazon_product_reviews.csv")

# Getting first 5 rows data 
df.head()
# Checking nlp pipeline 
nlp.pipe_names
# Getting all basic information of dataset
df.info()
# Checking the shape of dataset
df.shape
# Checking for any null values in the dataset
df.isnull().sum()
# Getting only reviews.text column from the dataset
df = df[['reviews.text']]
# Droping any null values from selected column
df.dropna(inplace=True)
# Checking for any null values in the column
df.isnull().sum()
# Getting the shape of selected column data
df.shape
#  Defining preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)
# Clean data after preprocessing
clean_data['reviews.text'] = clean_data['reviews.text'].apply(preprocess_text)
# Importing TextBlob
from textblob import TextBlob
# Defining Sentimental analysis function
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.25:
       sentiment = 'positive'
    elif polarity < -0.25:
      sentiment = 'negative'
    else:
         sentiment = 'neutral'
    return sentiment, polarity
# Selected sample reviews for sentiment analysis 
sample_reviews = [
    "This product so far has not disappointed. My children love to use it and I like the ability to monitor control what content they see with ease.",
    "This amazon fire 8 inch tablet is the perfect size. I purchased it for my husband so that he has a bigger screen than just his phone. He had gotten me one a few years ago so I knew it would be a good purchase."
]
# Defining two empty lists to store the sentiment and polarity score values 
sentiments = []
polarity_score = []

# Applying function of sentiment analysis
for review in sample_reviews:
    preprocessed_review = preprocess_text(review)
    sentiment,polarity = sentiment_analysis(preprocessed_review)
    sentiments.append(sentiment)
    polarity_score.append(polarity)
    # Printing result
    print(sentiment, polarity)
    # Selecting specific reviews for similarity check
review1 = df['reviews.text'].iloc[0]
review2 = df['reviews.text'].iloc[1]

# Creating spaCy doc objects
doc1 = nlp(review1)
doc2 = nlp(review2)

# Calculating similarity
similarity = doc1.similarity(doc2)
similarity