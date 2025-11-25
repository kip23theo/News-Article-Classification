📰ABSTRACT

Fake news has become a major challenge in the digital era, spreading rapidly through social media and online platforms. Identifying misleading information manually is difficult, time-consuming, and unreliable.
This project presents a Machine Learning–based Fake News Detection System that analyzes the textual content of a news article and classifies it as Real or Fake.
Using the Fake and Real News Dataset from Kaggle, the system applies text preprocessing, TF-IDF vectorization, and Logistic Regression classification. A Streamlit web interface allows users to paste news articles, analyze them in real-time, and view confidence scores and keyword explanations.
This tool demonstrates an effective, lightweight approach to fake-news classification with an intuitive newspaper-style UI.


📘 PROJECT OVERVIEW

This project is an end-to-end Fake News Detection system built under the folder news_classifier.
The workflow includes:

Dataset Import (Fake.csv + True.csv)

Data Cleaning

Model Training

Fake-News Classifier UI (Streamlit)

Prediction + Confidence + Explainability

The final output is a fully functioning web app where users can test any article for authenticity.


🧪 METHODOLOGY
1️⃣ Data Collection

Dataset: Fake and Real News Dataset (Kaggle)

Creator: George McIntire

Files: Fake.csv (fake articles), True.csv (real articles)

2️⃣ Data Cleaning (data_clean.py)

Cleaning steps:

Lowercasing text

Removing punctuation

Removing stopwords

Keeping only text and label columns

Saving processed data as cleaned_news.csv

3️⃣ Model Training (train_model.py)

Steps:

Loading cleaned dataset

Removing empty / NaN text

Splitting data into training/testing sets

Converting text → numerical vectors using TF-IDF (5000 features)

Training Logistic Regression classifier

Generating classification report

Saving trained model (model.pkl) and vectorizer (tfidf.pkl)

4️⃣ Deployment with Streamlit (app.py)

App features:

Modern newspaper-style UI

Text input area for articles

Real/Fake prediction

Confidence score

Top TF-IDF keyword table

Example news article buttons

Downloadable result file



SYSTEM ARCHITECTURE DIAGRAM 

               ┌────────────────────┐
               │  Fake.csv / True.csv│
               └──────────┬──────────┘
                          │
                          ▼
               ┌────────────────────┐
               │   Data Cleaning    │
               │ (remove stopwords, │
               │ punctuation, etc.) │
               └──────────┬──────────┘
                          │
                 cleaned_news.csv
                          │
                          ▼
               ┌────────────────────┐
               │   TF-IDF Vectorizer│
               └──────────┬──────────┘
                          │
                          ▼
               ┌────────────────────┐
               │ Logistic Regression│
               └──────────┬──────────┘
                          │
                model.pkl / tfidf.pkl
                          │
                          ▼
               ┌────────────────────┐
               │   Streamlit App    │
               │ (User pastes text) │
               └──────────┬──────────┘
                          │
                          ▼
               ┌────────────────────┐
               │Prediction: Real/Fake│
               │+ Confidence Score   │
               └────────────────────┘


DATA FLOW DIAGRAM

User Input → Preprocessing → TF-IDF Transformation → Model Prediction → Output Display (Real/Fake + Confidence + Keywords)



📝 CONCLUSION

This project demonstrates how Machine Learning can effectively classify news articles as real or fake using text-based analysis.
By combining TF-IDF vectorization, Logistic Regression, and Streamlit, the system delivers:

Real-time predictions

A clean and friendly UI

Explainability through keyword importance

A practical demonstration of NLP + ML workflow

The lightweight model ensures fast processing while maintaining good accuracy.


FUTURE WORK

Upgrade to transformer models (BERT, RoBERTa)

Add multilingual news detection

Implement URL/content extraction

Add PDF / DOCX / image-to-text upload

Deploy on the cloud (Streamlit Cloud / Render / HuggingFace Spaces)

Add fake-news probability scoring system