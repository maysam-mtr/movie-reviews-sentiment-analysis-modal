import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re # regular expressions library for pattern matching
from bs4 import BeautifulSoup
import nltk #nl toolkit for text processing
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords

# Load Dataset
df = pd.read_csv('./IMDB_Dataset.csv')

# 50,000 records

print(df.head())
print(df.tail())
#                                                review sentiment
# 0  One of the other reviewers has mentioned that ...  positive
# 1  A wonderful little production. <br /><br />The...  positive
# 2  I thought this was a wonderful way to spend ti...  positive
# 3  Basically there's a family where a little boy ...  negative
# 4  Petter Mattei's "Love in the Time of Money" is...  positive
#                                                    review sentiment
# 49995  I thought this movie did a down right good job...  positive
# 49996  Bad plot, bad dialogue, bad acting, idiotic di...  negative
# 49997  I am a Catholic taught in parochial elementary...  negative
# 49998  I'm going to have to disagree with the previou...  negative
# 49999  No one expects the Star Trek movies to be high...  negative

# Check for nan values
print(df.isna().sum())
# review       0
# sentiment    0

# check for imbalance
print(df['sentiment'].value_counts())
# sentiment
# positive    25000
# negative    25000
# Name: count, dtype: int64

#check for duplicates
print(df.duplicated().sum())
# 418 duplicates

#Drop duplicates
df.drop_duplicates(inplace=True)

# Encode categorical variable sentiment
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def preprocess_text(text):
    #change text to lowercase
    text = text.lower()

    #remove special characters and digits
    text = re.sub(r'\d+', '', text) # to remove digits
    #text = re.sub(r'[^\w\s]', '', text)  # to remove special characters

    text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters (keeping letters, numbers, and spaces)
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text) # Remove URLs
    text = text.strip() # Strip leading/trailing whitespaces

    #tokenize the text
    tokens = nltk.word_tokenize(text)

    return tokens


def clean_text(text):
    tokens = preprocess_text(text)

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    #perform lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    clean_text = ' '.join(lemmatized_tokens)
    return clean_text

# Apply the cleaning function to the 'review' column
df['review'] = df['review'].apply(clean_text)

# Display the first few rows of the cleaned reviews
print(df[['review']].head())
#                                              review                                     cleaned_review
#0  One of the other reviewers has mentioned that ...  one reviewer mentioned watching 1 oz episode y...
#1  A wonderful little production. <br /><br />The...  wonderful little production filming technique ...
#2  I thought this was a wonderful way to spend ti...  thought wonderful way spend time hot summer we...
#3  Basically there's a family where a little boy ...  basically there family little boy jake think t...
#4  Petter Mattei's "Love in the Time of Money" is...  petter matteis love time money visually stunni...

# Save the cleaned dataset if needed
df.to_csv('cleaned_reviews.csv', index=False)