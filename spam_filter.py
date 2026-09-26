import pandas as pd

# Load the dataset
df = pd.read_csv("spam_dataset.csv")

# -------------------------
# 1. Basic dataset info
# -------------------------

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["is_spam"].value_counts())

# -------------------------
# 2. Check for missing values
# -------------------------

print("\nMissing values:")
print(df.isnull().sum())

# -------------------------
# 3. Check for duplicate messages
# -------------------------

print("\nDuplicate rows:", df.duplicated().sum())

# -------------------------
# 4. Show a few examples
# -------------------------

print("\nFirst 5 messages:")

for i, row in df.head().iterrows():
    print(f"\nMessage {i + 1}")
    print("Spam:", row["is_spam"])
    print("Text:", row["message_content"])
