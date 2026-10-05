import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import f1_score


# ============================================================
# 1. ДАННЫЕ
# ============================================================

df = pd.read_csv('political_corpus/features.csv')

drop_cols = [
    'quadrant',
    'segment_id',
    'text_id',
    'segment_number',
    'segment_count',
    'author',
    'filename',
    'work',
    'genre',
    'style',
    'split',
    'file'
]

X = df.drop(columns=drop_cols).values

le = LabelEncoder()
y = le.fit_transform(df['quadrant'])

train_mask = df['split'] == 'train'
test_mask = df['split'] == 'test'

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

groups_train = df.loc[train_mask, 'text_id'].values


# ============================================================
# 2. ПРОВЕРКИ
# ============================================================

print("train text_id:", df.loc[train_mask, 'text_id'].nunique())
print("train классы:", pd.Series(y_train).value_counts().to_dict())
print("test  классы:", pd.Series(y_test).value_counts().to_dict())


# ============================================================
# 3. КРОСС-ВАЛИДАЦИЯ
# ============================================================

CV = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)


def evaluate(model, name):

    f1 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV,
        groups=groups_train,
        scoring='f1_macro'
    )

    print(
        f"\n[{name}] CV f1_macro: {f1.mean():.3f} "
        f"({', '.join(f'{s:.2f}' for s in f1)})"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(
        f"[{name}] Test f1_macro: "
        f"{f1_score(y_test, y_pred, average='macro'):.3f}"
    )


# ============================================================
# 4. DECISION TREE
# ============================================================

def demo_decision_tree():

    print("\n1. DECISION TREE")

    evaluate(
        DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=2,
            random_state=0
        ),
        "tree"
    )


# ============================================================
# 5. RANDOM FOREST
# ============================================================

def demo_random_forest():

    print("\n2. RANDOM FOREST")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=0
    )

    evaluate(model, "forest")


# ============================================================
# 6. SVM
# ============================================================

def demo_svm():

    print("\n3. SVM")

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=0
        )
    )

    evaluate(model, "svm")


# ============================================================
# 7. SOFTMAX ДЛЯ SVM
# ============================================================

def softmax(scores):

    scores = np.asarray(scores)

    scores = scores - np.max(scores)

    exp_scores = np.exp(scores)

    return exp_scores / exp_scores.sum()


# ============================================================
# 8. КООРДИНАТЫ ПОЛИТИЧЕСКОГО КОМПАСА
# ============================================================

def probabilities_to_point(probabilities):

    """
    ПОЛИТИЧЕСКИЙ КОМПАС

                    ЛЕВО                  ПРАВО

             Q2 КОММУНИСТЫ          Q3 МОНАРХИСТЫ
                (-1, +1)               (+1, +1)

             Q1 АНПРИМ               Q4 АНКАП
                (-1, -1)               (+1, -1)


    Q1 = нижний левый
    Q2 = верхний левый
    Q3 = верхний правый
    Q4 = нижний правый
    """

    q1, q2, q3, q4 = probabilities

    # X:
    #
    # Q1 = -1
    # Q2 = -1
    # Q3 = +1
    # Q4 = +1

    x = (
        -q1
        -q2
        +q3
        +q4
    )

    # Y:
    #
    # Q1 = -1
    # Q2 = +1
    # Q3 = +1
    # Q4 = -1

    y = (
        -q1
        +q2
        +q3
        -q4
    )

    return x, y


# ============================================================
# 9. ASCII-КОМПАС
# ============================================================

def print_compass(rf_point, svm_point, mean_point):

    rf_x, rf_y = rf_point
    svm_x, svm_y = svm_point

    mean_x, mean_y = mean_point

    WIDTH = 65
    HEIGHT = 27

    min_x = -1.15
    max_x = 1.15

    min_y = -1.15
    max_y = 1.15

    grid = [
        [' ' for _ in range(WIDTH)]
        for _ in range(HEIGHT)
    ]

    # ========================================================
    # КООРДИНАТЫ → ASCII
    # ========================================================

    def to_grid(x, y):

        gx = int(
            (x - min_x)
            / (max_x - min_x)
            * (WIDTH - 1)
        )

        gy = int(
            (max_y - y)
            / (max_y - min_y)
            * (HEIGHT - 1)
        )

        gx = max(0, min(WIDTH - 1, gx))
        gy = max(0, min(HEIGHT - 1, gy))

        return gx, gy

    # ========================================================
    # ОСИ
    # ========================================================

    zero_x, zero_y = to_grid(0, 0)

    for x in range(WIDTH):
        grid[zero_y][x] = '─'

    for y in range(HEIGHT):
        grid[y][zero_x] = '│'

    grid[zero_y][zero_x] = '┼'

    # ========================================================
    # ОБЛАСТЬ РАСХОЖДЕНИЯ RF / SVM
    # ========================================================

    area_min_x = min(rf_x, svm_x)
    area_max_x = max(rf_x, svm_x)

    area_min_y = min(rf_y, svm_y)
    area_max_y = max(rf_y, svm_y)

    # Небольшая область даже при близких точках

    padding_x = max(
        abs(rf_x - svm_x) * 0.35,
        0.025
    )

    padding_y = max(
        abs(rf_y - svm_y) * 0.35,
        0.025
    )

    area_min_x -= padding_x
    area_max_x += padding_x

    area_min_y -= padding_y
    area_max_y += padding_y

    for gy in range(HEIGHT):

        for gx in range(WIDTH):

            x = (
                min_x
                + gx / (WIDTH - 1)
                * (max_x - min_x)
            )

            y = (
                max_y
                - gy / (HEIGHT - 1)
                * (max_y - min_y)
            )

            if (
                area_min_x <= x <= area_max_x
                and area_min_y <= y <= area_max_y
            ):

                if grid[gy][gx] == ' ':
                    grid[gy][gx] = '░'

    # ========================================================
    # ТОЧКИ
    # ========================================================

    rf_gx, rf_gy = to_grid(
        rf_x,
        rf_y
    )

    svm_gx, svm_gy = to_grid(
        svm_x,
        svm_y
    )

    mean_gx, mean_gy = to_grid(
        mean_x,
        mean_y
    )

    # RF

    grid[rf_gy][rf_gx] = 'R'

    # SVM

    grid[svm_gy][svm_gx] = 'S'

    # Средняя точка

    grid[mean_gy][mean_gx] = 'M'

    # ========================================================
    # ЗАГОЛОВОК
    # ========================================================

    print()
    print("=" * 75)
    print("ПОЛИТИЧЕСКИЙ КОМПАС")
    print("=" * 75)

    print()

    print(
        "       Q2 КОММУНИСТЫ"
        + " " * 25
        + "Q3 МОНАРХИСТЫ"
    )

    print()

    # ========================================================
    # САМАЯ КАРТА
    # ========================================================

    for row in grid:
        print(''.join(row))

    print()

    print(
        "       Q1 АНПРИМ"
        + " " * 35
        + "Q4 АНКАП"
    )

    print()

    # ========================================================
    # ЛЕГЕНДА
    # ========================================================

    print("R = Random Forest")
    print("S = SVM")
    print("M = средняя точка RF + SVM")
    print("░ = область расхождения моделей")

    print()

    # ========================================================
    # КООРДИНАТЫ
    # ========================================================

    print(
        f"Random Forest : "
        f"({rf_x:+.3f}, {rf_y:+.3f})"
    )

    print(
        f"SVM           : "
        f"({svm_x:+.3f}, {svm_y:+.3f})"
    )

    print(
        f"Средняя точка : "
        f"({mean_x:+.3f}, {mean_y:+.3f})"
    )

    print()

    print(
        f"Расхождение X : "
        f"{abs(rf_x - svm_x):.3f}"
    )

    print(
        f"Расхождение Y : "
        f"{abs(rf_y - svm_y):.3f}"
    )


# ============================================================
# 10. РАСЧЁТ ПОЛИТИЧЕСКОГО КОМПАСА
# ============================================================

def political_compass():

    print()
    print("=" * 75)
    print("РАСЧЁТ ПОЛИТИЧЕСКОГО КОМПАСА")
    print("=" * 75)

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    forest = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=0
    )

    forest.fit(
        X_train,
        y_train
    )

    rf_probabilities = forest.predict_proba(
        X_test
    )

    rf_mean_probabilities = (
        rf_probabilities.mean(axis=0)
    )

    # ========================================================
    # SVM
    # ========================================================

    svm = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=0
        )
    )

    svm.fit(
        X_train,
        y_train
    )

    svm_scores = svm.decision_function(
        X_test
    )

    svm_probabilities = np.array([
        softmax(row)
        for row in svm_scores
    ])

    svm_mean_probabilities = (
        svm_probabilities.mean(axis=0)
    )

    # ========================================================
    # СОПОСТАВЛЯЕМ КЛАССЫ
    # ========================================================

    rf_prob = {}

    for class_number, probability in zip(
        forest.classes_,
        rf_mean_probabilities
    ):

        label = str(
            le.inverse_transform(
                [class_number]
            )[0]
        ).lower()

        rf_prob[label] = probability

    svm_prob = {}

    for class_number, probability in zip(
        svm[-1].classes_,
        svm_mean_probabilities
    ):

        label = str(
            le.inverse_transform(
                [class_number]
            )[0]
        ).lower()

        svm_prob[label] = probability

    # ========================================================
    # ПОЛУЧЕНИЕ Q1-Q4
    # ========================================================

    def get_probability(prob_dict, q):

        # Нормальный вариант

        if q in prob_dict:
            return prob_dict[q]

        # На случай Q1 вместо q1

        if q.upper() in prob_dict:
            return prob_dict[q.upper()]

        return 0.0

    rf_values = [
        get_probability(rf_prob, 'q1'),
        get_probability(rf_prob, 'q2'),
        get_probability(rf_prob, 'q3'),
        get_probability(rf_prob, 'q4')
    ]

    svm_values = [
        get_probability(svm_prob, 'q1'),
        get_probability(svm_prob, 'q2'),
        get_probability(svm_prob, 'q3'),
        get_probability(svm_prob, 'q4')
    ]

    # ========================================================
    # НОРМАЛИЗАЦИЯ
    # ========================================================

    rf_sum = sum(rf_values)

    if rf_sum > 0:

        rf_values = [
            value / rf_sum
            for value in rf_values
        ]

    svm_sum = sum(svm_values)

    if svm_sum > 0:

        svm_values = [
            value / svm_sum
            for value in svm_values
        ]

    # ========================================================
    # ТАБЛИЦА ВЕРОЯТНОСТЕЙ
    # ========================================================

    print()
    print("СРЕДНИЕ ОЦЕНКИ ПО TEST")
    print()

    print(
        f"{'Класс':<8}"
        f"{'Random Forest':>18}"
        f"{'SVM':>18}"
    )

    print("-" * 46)

    for i, q in enumerate(
        ['Q1', 'Q2', 'Q3', 'Q4']
    ):

        print(
            f"{q:<8}"
            f"{rf_values[i]:>17.1%}"
            f"{svm_values[i]:>17.1%}"
        )

    # ========================================================
    # КООРДИНАТЫ RF
    # ========================================================

    rf_point = probabilities_to_point(
        rf_values
    )

    # ========================================================
    # КООРДИНАТЫ SVM
    # ========================================================

    svm_point = probabilities_to_point(
        svm_values
    )

    # ========================================================
    # СРЕДНЯЯ ТОЧКА
    # ========================================================

    mean_point = (
        (rf_point[0] + svm_point[0]) / 2,
        (rf_point[1] + svm_point[1]) / 2
    )

    # ========================================================
    # ASCII-КОМПАС
    # ========================================================

    print_compass(
        rf_point,
        svm_point,
        mean_point
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    demo_decision_tree()

    demo_random_forest()

    demo_svm()

    political_compass()