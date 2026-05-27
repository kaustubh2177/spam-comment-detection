import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("youtube_spam.csv")
data = data[['CONTENT', 'CLASS']]
data.columns = ['text', 'label']

stop_words = set(stopwords.words('english'))

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(data['clean_text'])

spam_vectors = tfidf_matrix[data['label'] == 1]

# Prediction function
def predict_spam(comment, threshold=0.3):
    comment = preprocess(comment)
    comment_vector = vectorizer.transform([comment])
    similarity = cosine_similarity(comment_vector, spam_vectors)
    score = similarity.max()
    return "Spam" if score > threshold else "Not Spam"

# Test
print(predict_spam("Buy followers cheap click link now"))
