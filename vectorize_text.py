import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from clean_text import clean_text

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], ignore_index=True)

# Clean text
data["text"] = data["text"].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(data["text"])
y = data["label"]

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)