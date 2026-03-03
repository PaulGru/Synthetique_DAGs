import data_nlp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

def train_base_classifier():
    print("Loading data...")
    texts, labels = data_nlp.load_sms_spam_dataset()
    
    print("Training TF-IDF + Logistic Regression base model...")
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(solver='liblinear'))
    model.fit(texts, labels)
    
    print("Accuracy:", model.score(texts, labels))
    
    # Save model
    joblib.dump(model, 'base_spam_model.pkl')
    print("Model saved to base_spam_model.pkl")

if __name__ == "__main__":
    train_base_classifier()
