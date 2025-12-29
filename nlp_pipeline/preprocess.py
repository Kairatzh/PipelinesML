"""
Text preprocessing module.
- Cleaning
- Tokenization
- Stopwords removal
- Lemmatization (spaCy)
"""

import re
import nltk
import spacy

from nltk.corpus import stopwords

# Download once:
# nltk.download('stopwords')
# spacy.cli.download("ru_core_news_sm")

STOPWORDS = set(stopwords.words('russian'))
nlp = spacy.load("ru_core_news_sm")

def clean_text(text: str) -> str:
    """
    Очистка текста от шумов.
    """
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text: str) -> str:
    """
    Полный цикл препроцессинга:
      1. очистка
      2. токенизация (spaCy)
      3. лемматизация
      4. удаление стоп-слов
    """
    text = clean_text(text)
    doc = nlp(text)

    tokens = [
        token.lemma_ for token in doc 
        if token.lemma_ not in STOPWORDS and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

