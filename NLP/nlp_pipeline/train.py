"""
Model training pipeline:
- Load dataset
- Preprocess text
- Convert to features
- Train classifier
- Save model
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocess import preprocess
from features import get_tfidf

DATA_PATH = "data/reviews.csv"
MODEL_PATH = "model/classifier.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

df = pd.read_csv(DATA_PATH)

df["clean"] = df["text"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer = get_tfidf()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model saved to:", MODEL_PATH)
