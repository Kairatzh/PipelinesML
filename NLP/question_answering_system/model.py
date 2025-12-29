"""
Question Answering model wrapper.
"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

MODEL_NAME = "deepset/roberta-base-squad2"  # рус/англ зависит от задач
# Для русского можно использовать ai-forever/ruT5-base для генерации ответа

class QASystem:
    def __init__(self):
        self.qa = pipeline("question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME)

    def answer(self, question: str, context: str):
        result = self.qa(question=question, context=context)
        return result["answer"], result.get("score", None)
