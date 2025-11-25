import pandas as pd
import string
from nltk.corpus import stopwords

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Label data
fake['label'] = 0
real['label'] = 1

# Combine them
data = pd.concat([fake, real], ignore_index=True)
data = data[['text', 'label']]

# Clean text function
stop = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # lowercase
    text = ''.join([c for c in text if c not in string.punctuation])  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop])  # remove stopwords
    return text

# Apply cleaning
data['text'] = data['text'].apply(clean_text)

# Save cleaned file
data.to_csv("cleaned_news.csv", index=False)

print("Cleaning complete! File saved as cleaned_news.csv")
