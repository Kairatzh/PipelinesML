"""
Feature extraction: TF-IDF and BoW.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_tfidf():
    """Возвращает TF-IDF векторизатор."""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2)
    )

def get_bow():
    """Возвращает Bag of Words векторизатор."""
    return CountVectorizer(
        max_features=5000
    )
