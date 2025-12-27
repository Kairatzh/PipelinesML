"""
Breast Cancer Classification Pipeline
-------------------------------------

Этот скрипт демонстрирует полный ML-пайплайн классического машинного обучения:
- загрузка данных
- препроцессинг с ColumnTransformer и Pipeline
- обучение базовых моделей
- тюнинг гиперпараметров (GridSearchCV)
- ансамблирование (Voting, Stacking)
- финальная оценка метрик

Цель: показать структуру продакшен-ориентированного классического ML.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold, 
    GridSearchCV
    )
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    VotingClassifier, 
    StackingClassifier
    )
from sklearn.svm import SVC

#Глобальный seed для воспроизводимости
RANDOM_STATE = 1


# ============================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================

"""
load_breast_cancer возвращает структурированный датасет от sklearn.
as_frame=True позволяет получить его в виде pandas.DataFrame.
"""
data = load_breast_cancer(as_frame=True)
X = data.data          # признаки
y = data.target        # целевая переменная (0 или 1)
num_features = X.columns.tolist()  # список числовых фичей (все числовые)


# ============================
# 2. TRAIN/TEST SPLIT
# ============================

"""
stratify=y гарантирует, что доля классов сохранится в train и test.
Это важно при несбалансированных задачах классификации.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================
# 3. ПРЕПРОЦЕССИНГ
# ============================

"""
numeric_transformer — пайплайн для обработки числовых признаков.

В данном примере:
- StandardScaler нормализует признаки: (x - mean) / std
"""
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

"""
ColumnTransformer позволяет независимо обрабатывать разные группы колонок.
В нашем случае — только числовые.
"""
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features)
    ]
)


# ============================
# 4. БАЗОВЫЕ МОДЕЛИ
# ============================

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1  # использование всех ядер процессора
)

gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

ada = AdaBoostClassifier(random_state=RANDOM_STATE)

svc = SVC(probability=True, random_state=RANDOM_STATE)


# ============================
# 5. PIPELINES
# ============================

"""
Pipeline связывает препроцессинг и модель в одну сущность,
чтобы дальше их можно было обучать одним вызовом fit().
"""

pipe_lr = Pipeline([("prep", preprocessor), ("model", log_reg)])
pipe_rf = Pipeline([("prep", preprocessor), ("model", rf)])
pipe_gb = Pipeline([("prep", preprocessor), ("model", gb)])


# ============================
# 6. ТЮНИНГ ГИПЕРПАРАМЕТРОВ
# ============================

"""
Grid Search по параметрам для GradientBoosting.
Параметры передаются с префиксом model__ потому что находятся внутри Pipeline.
"""

param_grid_gb = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_gb = GridSearchCV(
    estimator=pipe_gb,
    param_grid=param_grid_gb,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_  # оптимальный пайплайн


# ============================
# 7. АНСАМБЛИ
# ============================

"""
Soft Voting — усреднение вероятностей, увеличивает устойчивость к переобучению.
"""
voting = VotingClassifier(
    estimators=[
        ("lr", pipe_lr),
        ("rf", pipe_rf),
        ("gb", best_gb)
    ],
    voting="soft",
    n_jobs=-1
)

"""
Stacking — мета-модель обучается на предсказаниях базовых моделей.
"""
stacking = StackingClassifier(
    estimators=[
        ("lr", pipe_lr),
        ("rf", pipe_rf),
        ("gb", pipe_gb)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv,
    n_jobs=-1
)


# ============================
# 8. ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ
# ============================

models = {
    "LogReg": pipe_lr,
    "RandomForest": pipe_rf,
    "BestGB": best_gb,
    "Voting": voting,
    "Stacking": stacking
}

for name, model in models.items():
    model.fit(X_train, y_train)  # тренируем каждую модель


# ============================
# 9. МЕТРИКИ И ОЦЕНКА
# ============================

def evaluate(model, X_test, y_test):
    """
    Выполняет оценку модели по Accuracy и ROC-AUC.

    Parameters
    ----------
    model : sklearn.BaseEstimator
        Обученная модель или Pipeline.
    X_test : pd.DataFrame
        Тестовые признаки.
    y_test : pd.Series
        Истинные метки класса.

    Returns
    -------
    dict :
        accuracy : float
        roc_auc : float
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


results = {name: evaluate(model, X_test, y_test) for name, model in models.items()}

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print("\n=== MODEL PERFORMANCE ===")
print(results_df)

best_model = results_df.index[0]
print(f"\nЛучший модель по ROC-AUC: {best_model}\n")
print(classification_report(y_test, models[best_model].predict(X_test)))
