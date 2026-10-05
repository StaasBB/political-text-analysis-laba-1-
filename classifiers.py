import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import f1_score

# --- 1. Пути к файлам ---
TRAIN_CSV = 'political_corpus/train_features.csv'
TEST_CSV  = 'political_corpus/test_features.csv'

# --- 2. Признаки, которые не должны попасть в X ---
drop_cols = ['quadrant', 'segment_id', 'text_id', 'segment_number', 'segment_count',
             'author', 'filename', 'work', 'genre', 'style', 'split', 'file']

# --- 3. Загрузка train и test отдельно ---
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

feature_cols = [c for c in df_train.columns if c not in drop_cols]

missing_in_test = [c for c in feature_cols if c not in df_test.columns]
if missing_in_test:
    raise ValueError(
        f"В {TEST_CSV} отсутствуют признаки: {missing_in_test[:10]}"
        + (" ..." if len(missing_in_test) > 10 else "")
    )

# --- 4. X и y ---
X_train = df_train[feature_cols].values
X_test  = df_test[feature_cols].values 

le = LabelEncoder()

y_train = le.fit_transform(df_train['quadrant'])
y_test  = le.transform(df_test['quadrant'])

groups_train = df_train['text_id'].values

# --- 5. Проверки ---
print("train text_id:", df_train['text_id'].nunique())
print("train классы:", pd.Series(y_train).value_counts().to_dict())
print("test  классы:", pd.Series(y_test).value_counts().to_dict())
print("признаков:", len(feature_cols))
print("X_train:", X_train.shape, " X_test:", X_test.shape)

# --- 6. Кросс-валидация ---
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

def evaluate(model, name):
    f1 = cross_val_score(model, X_train, y_train,
                         cv=CV, groups=groups_train, scoring='f1_macro')
    print(f"\n[{name}] CV f1_macro: {f1.mean():.3f} "
          f"({', '.join(f'{s:.2f}' for s in f1)})")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"[{name}] Test f1_macro: {f1_score(y_test, y_pred, average='macro'):.3f}")

def get_probabilities(model, vec):
    """
    Возвращает словарь {Q1: p1, Q2: p2, Q3: p3, Q4: p4} для одного вектора.
    Для моделей с predict_proba берём оттуда.
    Для LinearSVC — softmax от decision_function.
    """
    vec = vec.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = model.classes_
        return {le.inverse_transform([c])[0]: float(p) for c, p in zip(classes, proba)}

    if hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        exp = np.exp(scores - scores.max())
        proba = exp / exp.sum()
        classes = model.classes_
        return {le.inverse_transform([c])[0]: float(p) for c, p in zip(classes, proba)}

    return None

def print_test_predictions(models, X_test, y_test, le, df_test):
    """
    Для каждого тестового сегмента печатает предсказание каждой модели,
    истинный класс (если есть) и распределение вероятностей.
    """
    print("\n" + "=" * 90)
    print("ПРЕДСКАЗАНИЯ ПО ТЕСТОВЫМ СЕГМЕНТАМ")
    print("=" * 90)

    true_labels = le.inverse_transform(y_test)

    for i in range(len(X_test)):
        vec = X_test[i]
        seg_id = df_test.iloc[i].get("segment_id", f"row {i+1}")
        true_label = true_labels[i]

        print(f"\n--- {seg_id}  (истина: {true_label}) ---")

        for name, model in models.items():
            pred_encoded = model.predict(vec.reshape(1, -1))[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            proba = get_probabilities(model, vec)

            if proba is None:
                print(f"  [{name:<6}] → {pred_label}")
                continue

            items = sorted(proba.items(), key=lambda kv: -kv[1])
            proba_str = "  ".join(f"{k}:{v:.2f}" for k, v in items)
            mark = "✓" if pred_label == true_label else "✗"
            print(f"  [{name:<6}] → {pred_label} {mark}   {proba_str}")

# --- 7. Модели ---
def demo_decision_tree():
    print("\n1. DECISION TREE")
    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=0)
    evaluate(model, "tree")
    return model

def demo_random_forest():
    print("\n2. RANDOM FOREST")
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                   random_state=0)
    evaluate(model, "forest")
    return model

def demo_svm():
    print("\n3. SVM")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight='balanced', random_state=0)
    )
    evaluate(model, "svm")
    return model

# --- 8. Запуск ---
if __name__ == "__main__":
    tree   = demo_decision_tree()
    forest = demo_random_forest()
    svm    = demo_svm()

    models = {"tree": tree, "forest": forest, "svm": svm}
    print_test_predictions(models, X_test, y_test, le, df_test)