import pandas as pd
from clean_text import clean_text

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Clean text column
data["text"] = data["text"].apply(clean_text)

# Show sample output
print(data["text"].head())
print("\nTotal rows:", len(data))