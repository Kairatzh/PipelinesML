"""
Inference for QA query: context.txt + question input.
"""

from preprocess import preprocess_context
from qa_model import QASystem

CONTEXT_PATH = "data/context.txt"

def load_context():
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def ask(question: str):
    raw = load_context()
    proc = preprocess_context(raw)

    qa = QASystem()
    answer, score = qa.answer(question, proc["clean_text"])

    print("Вопрос:", question)
    print("Ответ:", answer)
    print("Уверенность:", round(score, 3))

if __name__ == "__main__":
    user_q = input("Введите вопрос: ")
    ask(user_q)
