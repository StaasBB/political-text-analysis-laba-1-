import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import f1_score

# --- 1. Данные ---
df = pd.read_csv('political_corpus/features.csv')

drop_cols = ['quadrant', 'segment_id', 'text_id', 'segment_number', 'segment_count',
             'author', 'filename', 'work', 'genre', 'style', 'split', 'file']
X = df.drop(columns=drop_cols).values

le = LabelEncoder()
y = le.fit_transform(df['quadrant'])

train_mask = df['split'] == 'train'
test_mask  = df['split'] == 'test'

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
groups_train = df.loc[train_mask, 'text_id'].values

# --- 2. Проверки ---
print("train text_id:", df.loc[train_mask, 'text_id'].nunique())
print("train классы:", pd.Series(y_train).value_counts().to_dict())
print("test  классы:", pd.Series(y_test).value_counts().to_dict())

# --- 3. Кросс-валидация ---
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

def evaluate(model, name):
    f1 = cross_val_score(model, X_train, y_train,
                         cv=CV, groups=groups_train, scoring='f1_macro')
    print(f"\n[{name}] CV f1_macro: {f1.mean():.3f} "
          f"({', '.join(f'{s:.2f}' for s in f1)})")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"[{name}] Test f1_macro: {f1_score(y_test, y_pred, average='macro'):.3f}")

# --- 4. Модели ---
def demo_decision_tree():
    print("\n1. DECISION TREE")
    evaluate(DecisionTreeClassifier(max_depth=10, min_samples_leaf=2,
                                    random_state=0), "tree")

def demo_random_forest():
    print("\n2. RANDOM FOREST")
    evaluate(RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    random_state=0), "forest")

def demo_svm():
    print("\n3. SVM")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight='balanced', random_state=0)
    )
    evaluate(model, "svm")

if __name__ == "__main__":
    demo_decision_tree()
    demo_random_forest()
    demo_svm()