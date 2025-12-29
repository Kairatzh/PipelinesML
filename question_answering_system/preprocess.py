"""
Preprocessing pipeline for QA:
- Cleaning
- spaCy NER & Lemmatization
- Sentence Segmentation (for faster answer search)
"""

import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize

# One-time setup:
# nltk.download('punkt')
# spacy.cli.download("ru_core_news_sm")

nlp = spacy.load("ru_core_news_sm")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_context(text: str):
    """
    Полный препроцессинг контекста:
    - Очистка
    - Разбиение на предложения
    - NER (выделение сущностей)
    """
    text = clean_text(text)
    sentences = sent_tokenize(text, language='russian')
    doc = nlp(text)

    ner = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        "clean_text": text,
        "sentences": sentences,
        "entities": ner
    }
