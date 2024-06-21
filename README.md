# Sentiment_Analysis on Consumer Reviews of Amazon Products
Introduction
This project focuses on performing sentiment analysis on Consumer Reviews of Amazon products. The goal is to classify the sentiment of reviews as positive, negative, or neutral. By analyzing these sentiments, businesses can gain insights into customer satisfaction and product performance.
Dataset: Consumer Reviews of Amazon Products
1. Description of the Dataset
The dataset used for the sentiment analysis in this task is Consumer Reviews of Amazon Products. It is
cosisted of 21 columns and 34660 rows which includes the product related information, product
reviews,ratings and customer related information. The most important column which is used for
sentiment analysis is 'review.text', in which reviews by customers is available.
2. Preprocessing Steps
To perform the sentiment analysis the dataset needs to be prepared for it. For this purpose I have done
several preprocessing steps which were as follow:
1. Importing Libraries:
Libraries were imported to perform different procedures on the dataset.
2. Loading the Dataset:
The dataset was loaded into a Pandas DataFrame for easier manipulation and analysis.
3. Removing Missing Values:
Any missing values in the `reviews.text` column were removed using the `dropna` method.
4. Preprocessing
● Text Cleaning:
The review texts were cleaned to remove any unnecessary characters and whitespace. This included:
· Converting all text to lowercase to ensure uniformity.
· Removing punctuation, special characters, and numbers.
5. Tokenization and Stop words Removal:
Using the spaCy library, the reviews text was tokenized into individual words (tokens). Common
stopwords, which do not contribute to the sentiment, were removed.
6. Lemmatization:
Words were lemmatized to their base form to reduce the dimensionality of the data and ensure
consistency.
Sentiment Analysis Function
I defined the sentiment analysis function which evaluates the sentiment of the review and polarity score of
review.
The sentiment analysis model was implemented using the spaCy library combined with the TextBlob
extension to evaluate the sentiment polarity and of each review. The polarity score ranges from -1
(negative sentiment) to +1 (positive sentiment), with scores around 0 indicating neutral sentiment. In this
analysis
polarity > 0.25 indicates positive sentiment
polarity < -0.25 indicates negative sentiment
polarity between 0.25 and -0.25 indicates neutral sentiment
Sentiment Analysis Result:
Review1: "This product so far has not disappointed. My children love to use it and I like the ability to
monitor control what content they see with ease."
Polarity: -0.05000000000000001
Sentiment: Neutral
Review 2: "This amazon fire 8 inch tablet is the perfect size. I purchased it for my husband so that
he has a bigger screen than just his phone. He had gotten me one a few years ago so I knew it
would be a good purchase."
Polarity: 0.5666666666666667
Sentiment: Positive
Similarity: 0.5228083820301257
These results demonstrate the model's ability to classify the sentiment of reviews accurately based on
their polarity scores.
4. Insights into the Model's Strengths and Limitations
➢ Strengths:
Ease of Use:
I used spaCy and TextBlob libraries which provide easy-to-use APIs for natural language processing and
sentiment analysis.
Accuracy:
This model is capable of accurately identifying the sentiment of reviews in form of negative, positive and
neutral language.
Flexibility:
The model can be extended and fine-tuned with more sophisticated NLP techniques or additional data to
improve performance.
➢ Limitations:
Context Understanding:
The model may struggle in understanding the complex sentences or sarcasm and it effect on the accuracy
of the sentimental analysis.
Neutral language sentences may also become challenging to understand and classify correctly.
Preprocessing Dependency
The performance of the model depends upon the preprocessing steps. Inadequate cleaning of data or
tokenization can result in incorrect sentiment analysis
