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


# In[7]:


from sklearn.model_selection import train_test_split

# Split data: 80% for training, 20% for testing

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


print("Training set shape (X_train):", x_train.shape)
print("Training set (X_train):", x_train)
print("Testing set shape (X_test):", x_test.shape)
print("Testing set (X_test):", x_test)

print("Training labels shape (y_train):", y_train.shape)
print("Training labels (y_train):", y_train)
print("Testing labels shape (y_test):", y_test.shape)
print("Testing labels (y_test):", y_test)

