import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load cleaned data
data = pd.read_csv("cleaned_news.csv")

# FIX: Remove empty / NaN text
data = data.dropna(subset=['text'])
data = data[data['text'].str.strip() != ""]

X = data['text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numbers
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Test accuracy
preds = model.predict(X_test_tfidf)
print("Model Performance:")
print(classification_report(y_test, preds))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Training complete! Model saved as model.pkl and tfidf.pkl")
