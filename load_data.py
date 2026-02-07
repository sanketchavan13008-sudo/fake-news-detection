import pandas as pd

print("File started")

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Keep only required columns
data = data[["text", "label"]]

print("Final dataset shape:", data.shape)
print(data.head())