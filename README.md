# IMDB Movie Reviews Sentiment Analysis

![IMDB Logo](https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg)

### Sentiment Analysis of Movie Reviews using NLP & Machine Learning (Logistic Regression + Naive Bayes + TF-IDF)
---

## Project Overview

This project demonstrates how to classify **movie reviews** as **Positive** or **Negative** using **Natural Language Processing (NLP)** and **Machine Learning**. The pipeline includes data preprocessing, feature extraction with **TF-IDF**, training **Naive Bayes** and **Logistic Regression** models, evaluating performance, and predicting sentiment for new reviews.

---

## Dataset

The dataset is sourced from **NLTK's movie_reviews corpus**, containing **2,000 movie reviews** labeled as:

- **pos** → Positive sentiment  
- **neg** → Negative sentiment  

**Dataset Distribution**:

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive  | 1000  | 50%        |
| Negative  | 1000  | 50%        |

---

## Steps in the Project

### 1. Import Libraries
Used Python libraries for data processing, NLP, and machine learning:

- `nltk` → Tokenization, lemmatization, stopwords  
- `string` → Punctuation removal  
- `sklearn` → TF-IDF vectorization, train-test split, classifiers, evaluation metrics  

### 2. Data Preprocessing
- Lowercased all text  
- Removed **punctuation** and **stopwords**  
- Tokenized text and applied **lemmatization**  
- Cleaned and prepared reviews for feature extraction  

Example:

**Original Review**:  
Mr. Bean, a bumbling security guard from England is sent to LA...

**Preprocessed Review**:  
mr bean bumbling security guard england sent la help grandiose homecoming masterpiece...

### 3. Feature Extraction
- Applied **TF-IDF Vectorization**:
  - Converts text into numerical features  
  - Used **unigrams and bigrams** (`ngram_range=(1,2)`)  
  - Maximum 5,000 features, filtered rare and common terms  

- Training features shape: `(1600, 5000)`  
- Testing features shape: `(400, 5000)`  

### 4. Train-Test Split
- Split dataset into **80% training** and **20% testing** sets  
- Stratified to maintain class balance  

### 5. Model Training
- **Naive Bayes (MultinomialNB)**
- **Logistic Regression (liblinear solver)**

Training accuracy:

| Model               | Training Accuracy |
|--------------------|-----------------|
| Naive Bayes         | 78.5%           |
| Logistic Regression | 82.5%           |

### 6. Evaluation
Test performance:

**Naive Bayes**

| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Negative   | 0.75      | 0.84   | 0.80     | 200     |
| Positive   | 0.82      | 0.72   | 0.77     | 200     |

**Logistic Regression**

| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Negative   | 0.82      | 0.82   | 0.82     | 200     |
| Positive   | 0.82      | 0.82   | 0.82     | 200     |

- Confusion matrices visualize correct and incorrect predictions  
- Observations:
  - Logistic Regression performs slightly better than Naive Bayes  
  - Model captures sentiment effectively for both positive and negative reviews  

### 7. Real-time Prediction
- Function `predict_sentiment_tfidf()` allows predicting sentiment of new reviews:

```python
Enter a review: This movie was absolutely fantastic. A masterpiece of cinema
Predicted sentiment: Positive, Confidence: 0.65

Enter a review: This movie was terrible. I did not enjoy it at all
Predicted sentiment: Negative, Confidence: 0.56
