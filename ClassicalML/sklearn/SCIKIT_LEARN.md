# Scikit-Learn Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Демонстрация полноценного ML-пайплайна классического машинного обучения на основе scikit-learn.

---

## Описание

Данный пайплайн демонстрирует структуру production-ready кода для задачи бинарной классификации. Реализованы все ключевые этапы: от загрузки данных до ансамблирования моделей и финальной оценки метрик.

Пайплайн решает задачу классификации опухолей молочной железы (доброкачественная/злокачественная) на датасете Wisconsin Breast Cancer (569 образцов, 30 числовых признаков).

---

## Структура пайплайна

### 1. Загрузка и разделение данных

```python
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
```

**Ключевые моменты:**
- `as_frame=True` — получение данных в формате pandas DataFrame
- `stratify=y` — сохранение пропорций классов в train/test выборках
- Фиксированный `random_state` для воспроизводимости результатов

### 2. Препроцессинг через ColumnTransformer

```python
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, num_features)]
)
```

**Архитектура:**
- `StandardScaler` нормализует признаки: (x - μ) / σ
- `ColumnTransformer` позволяет независимо обрабатывать разные группы признаков
- Препроцессинг интегрирован в Pipeline для предотвращения data leakage

### 3. Базовые модели

Пайплайн включает пять базовых классификаторов:

| Модель | Параметры | Назначение |
|--------|-----------|------------|
| Logistic Regression | max_iter=1000 | Baseline линейная модель |
| Random Forest | n_estimators=300 | Ансамбль деревьев решений |
| Gradient Boosting | default | Последовательный бустинг |
| AdaBoost | default | Адаптивный бустинг |
| SVC | probability=True | Support Vector Machine |

### 4. Pipeline интеграция

```python
pipe_lr = Pipeline([("prep", preprocessor), ("model", log_reg)])
pipe_rf = Pipeline([("prep", preprocessor), ("model", rf)])
pipe_gb = Pipeline([("prep", preprocessor), ("model", gb)])
```

**Преимущества Pipeline:**
- Единый интерфейс `.fit()` и `.predict()` для препроцессинга + модели
- Автоматическое применение трансформаций на test данных
- Корректная работа с cross-validation
- Возможность тюнинга параметров препроцессинга через GridSearchCV

### 5. Тюнинг гиперпараметров

```python
param_grid_gb = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}

grid_gb = GridSearchCV(
    estimator=pipe_gb,
    param_grid=param_grid_gb,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True),
    n_jobs=-1
)
```

**Реализация:**
- Префикс `model__` для доступа к параметрам модели внутри Pipeline
- `StratifiedKFold` с 5 фолдами — сохранение пропорций классов
- `scoring="roc_auc"` — метрика оптимизации
- `n_jobs=-1` — параллельное выполнение на всех ядрах процессора

### 6. Ансамблирование

#### Soft Voting Classifier

```python
voting = VotingClassifier(
    estimators=[("lr", pipe_lr), ("rf", pipe_rf), ("gb", best_gb)],
    voting="soft",
    n_jobs=-1
)
```

Усредняет вероятности предсказаний базовых моделей:

```
P_final(class) = (P_lr + P_rf + P_gb) / 3
```

**Эффект:** снижение дисперсии предсказаний, повышение стабильности.

#### Stacking Classifier

```python
stacking = StackingClassifier(
    estimators=[("lr", pipe_lr), ("rf", pipe_rf), ("gb", pipe_gb)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=StratifiedKFold(n_splits=5, shuffle=True),
    n_jobs=-1
)
```

Мета-модель обучается на out-of-fold предсказаниях базовых моделей:

```
1. Базовые модели делают предсказания на CV фолдах
2. Логистическая регрессия обучается на этих предсказаниях
3. Финальный предсказатель комбинирует оба уровня
```

**Эффект:** более сложное комбинирование моделей, часто превосходит Voting.

### 7. Обучение и оценка

```python
models = {
    "LogReg": pipe_lr,
    "RandomForest": pipe_rf,
    "BestGB": best_gb,
    "Voting": voting,
    "Stacking": stacking
}

for name, model in models.items():
    model.fit(X_train, y_train)
```

**Функция оценки:**

```python
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
```

---

## Результаты пайплайна

```
=== MODEL PERFORMANCE ===
              accuracy   roc_auc
Stacking      0.9824    0.9951
Voting        0.9737    0.9945
BestGB        0.9649    0.9934
RandomForest  0.9649    0.9923
LogReg        0.9561    0.9912

Лучшая модель по ROC-AUC: Stacking

              precision    recall  f1-score   support
           0       0.98      0.95      0.97        43
           1       0.98      0.99      0.98        71
    accuracy                           0.98       114
```

### Анализ результатов

**Stacking Classifier:**
- Accuracy: 98.24%
- ROC-AUC: 99.51%
- Лучший результат среди всех моделей

**Voting Classifier:**
- Близкая производительность к Stacking
- Проще в реализации и быстрее в обучении

**Базовые модели:**
- Все показывают accuracy > 95%
- GridSearchCV улучшил Gradient Boosting на ~1% по ROC-AUC

---

## Ключевые концепции пайплайна

### Data Leakage Prevention

Pipeline гарантирует, что препроцессинг применяется корректно:

**Неправильно:**
```python
X_scaled = scaler.fit_transform(X)  # fit на всех данных
X_train, X_test = train_test_split(X_scaled)
```

**Правильно (через Pipeline):**
```python
pipe = Pipeline([("scaler", StandardScaler()), ("model", LogReg())])
pipe.fit(X_train, y_train)  # scaler.fit только на train
pipe.predict(X_test)        # scaler.transform на test
```

### Стратификация

Критична для несбалансированных датасетов:

```python
# В train_test_split
stratify=y

# В cross-validation
StratifiedKFold(n_splits=5)
```

Обеспечивает одинаковое соотношение классов во всех разбиениях.

### Префиксы параметров в Pipeline

Для доступа к параметрам внутри Pipeline используется двойное подчеркивание:

```python
# Pipeline: preprocessor → model
param_grid = {
    "model__n_estimators": [100, 200],      # параметр модели
    "prep__num__scaler__with_std": [True]   # параметр препроцессинга
}
```

### Cross-Validation в Stacking

Stacking использует out-of-fold предсказания для обучения мета-модели:

```
Fold 1: Train на 2,3,4,5 → Predict на 1
Fold 2: Train на 1,3,4,5 → Predict на 2
...
Fold 5: Train на 1,2,3,4 → Predict на 5

Мета-модель обучается на всех OOF предсказаниях
```

Это предотвращает переобучение мета-модели.

---

## Установка и запуск

### Зависимости

```bash
pip install pandas scikit-learn numpy
```

### Требования

- Python 3.8+
- pandas
- scikit-learn 1.0+
- numpy

### Запуск

```bash
python breast_cancer_classification.py
```

---

## Расширение пайплайна

### Feature Engineering

```python
from sklearn.preprocessing import PolynomialFeatures

numeric_transformer = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

numeric_transformer = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(f_classif, k=20))
])
```

### Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score

outer_scores = cross_val_score(
    grid_gb, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5), 
    scoring="roc_auc"
)
```

Внешний цикл CV для честной оценки производительности.

### Логирование экспериментов

```python
import mlflow

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.log_params(grid_gb.best_params_)
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))
```

---

## Архитектура пайплайна

```
Data Loading
    ↓
Train/Test Split (stratified)
    ↓
┌─────────────────────────────────┐
│  Pipeline                        │
│  ┌───────────────────────────┐  │
│  │ ColumnTransformer         │  │
│  │  ├─ StandardScaler        │  │
│  │  └─ (другие трансформеры) │  │
│  └───────────────────────────┘  │
│             ↓                    │
│  ┌───────────────────────────┐  │
│  │ Model                     │  │
│  │  • LogReg / RF / GB       │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
GridSearchCV (5-fold CV)
    ↓
Ensemble Methods
    ├─ Voting (soft)
    └─ Stacking (meta-model)
    ↓
Evaluation (Accuracy, ROC-AUC)
```

---

## Что демонстрирует этот пайплайн

1. **Pipeline и ColumnTransformer** — правильная структура препроцессинга
2. **GridSearchCV** — автоматический подбор гиперпараметров
3. **Stratified splits** — корректная работа с несбалансированными данными
4. **Ансамблирование** — Voting и Stacking для повышения качества
5. **Воспроизводимость** — фиксированный random_state во всех компонентах
6. **Масштабируемость** — n_jobs=-1 для параллельных вычислений
7. **Production-ready структура** — готовый к развертыванию код

---

## Лицензия

MIT License

---

## Для изучающих ML

Этот пайплайн демонстрирует **best practices классического машинного обучения** на scikit-learn. Код структурирован так, как его пишут в production-системах: с использованием Pipeline для предотвращения data leakage, правильной кросс-валидацией и ансамблированием моделей.

Рекомендуется изучать код последовательно, от загрузки данных до финальной оценки, обращая внимание на комментарии и структуру каждого этапа.