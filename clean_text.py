import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# test
sample = "Breaking News!!! This is VERY important."
print(clean_text(sample))
