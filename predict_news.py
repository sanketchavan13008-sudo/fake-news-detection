import joblib
from clean_text import clean_text

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Enter a news article (press Enter twice to submit):")

# Take multiline input
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

news_text = " ".join(lines)

# Clean and vectorize
cleaned_text = clean_text(news_text)
vectorized_text = vectorizer.transform([cleaned_text])

# Prediction
# Prediction with probability
probability = model.predict_proba(vectorized_text)[0]

fake_prob = probability[0]
real_prob = probability[1]

print(f"\nFake probability: {fake_prob:.2f}")
print(f"Real probability: {real_prob:.2f}")

if real_prob > fake_prob:
    print("✅ Prediction: REAL NEWS")
else:
    print("🛑 Prediction: FAKE NEWS")
