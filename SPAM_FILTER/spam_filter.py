#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv("spam_dataset.csv")

# -------------------------
# 1. Basic dataset info
# -------------------------

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["is_spam"].value_counts())


# In[2]:


# -------------------------
# 2. Check for missing values
# -------------------------

print("\nMissing values:")
print(df.isnull().sum())



# In[3]:


# -------------------------
# 3. Check for duplicate messages
# -------------------------

print("\nDuplicate rows:", df.duplicated().sum())


# In[4]:


# -------------------------
# 4. Show a few examples
# -------------------------

print("\nFirst 5 messages:")

for i, row in df.head().iterrows():
    print(f"\nMessage {i + 1}")
    print("Spam:", row["is_spam"])
    print("Text:", row["message_content"])


# In[5]:


#----------------------------
# 5. PreProcessing Data Cleaning
#----------------------------

df = df.dropna(
    subset=["message_content", "is_spam"]
).copy()

print("\nAfter dropping missing values, dataset shape:", df.shape)

#------------------------------
# Remove exact duplicate rows
#------------------------------

df = df.drop_duplicates().copy()

# Clean whitespace from messages
df["message_content"] = (
    df["message_content"]
    .astype(str)
    .str.strip()
)


# Remove empty messages
df = df[df["message_content"] != ""].copy()

# Ensure labels are valid binary values
df["is_spam"] = pd.to_numeric(
    df["is_spam"], errors="coerce"
)

df = df[df["is_spam"].isin([0, 1])].copy()
df["is_spam"] = df["is_spam"].astype(int)

df = df.reset_index(drop=True)

print("Shape after cleaning:", df.shape)
print(df["is_spam"].value_counts())


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer (keeping the top 5,000 most common words)

vectorizer = TfidfVectorizer(max_features=5000)

# Convert text messages into numerical features (X) and assign labels (y)

x = vectorizer.fit_transform(df["message_content"])
y = df["is_spam"].values


print("Shape of feature matrix (X):", x.shape)
print("X - Matrix", x)
print("Shape of label vector (y):", y.shape)
print("y-label", y)

