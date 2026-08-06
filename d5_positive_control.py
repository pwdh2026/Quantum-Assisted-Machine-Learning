"""O8 positive control: does the D+ decision pipeline flag a KNOWN stable advantage?

Pair with known advantage: tuned RBF vs linear SVM (nonlinear data -> RBF stably better).
Same pipeline as O5/O6: per-fold paired delta, sigma_fold, 3 seeds, matched subsample
(churn 375) / full (glass). If pipeline flags 'alive' here but not for quantum pairs,
the decision logic is discriminative (closed loop).
"""
import sys

import numpy as np
import pandas as pd
import pennylane as qml  # noqa: F401 (env consistency)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = r"C:\Users\38277\.qwenworkcn\workspace\msg779bf8k5wk0rc\Quantum-Assisted-Machine-Learning\datasets"


def load_churn_sub():
    df = pd.read_csv(f"{BASE}/bank_churn.csv")
    X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)
    y = df["Exited"].values
    idx = np.random.RandomState(42).permutation(len(X))[:375]
    return X.values.astype(float)[idx], y[idx]


def load_glass():
    df = pd.read_csv(f"{BASE}/glass.csv")
    X = df.drop(columns=[df.columns[-1]]).values.astype(float)
    y = df[df.columns[-1]].values
    return X, y


def eval_pair(X, y, seed, pca_k=2):
    deltas = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        if pca_k > 0:
            p = PCA(pca_k).fit(sc.transform(X[tr]))
            Xtr, Xte = p.transform(sc.transform(X[tr])), p.transform(sc.transform(X[te]))
        else:
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lin = SVC(kernel="linear").fit(Xtr, y[tr])
        rbf = GridSearchCV(SVC(kernel="rbf"), {"C": [0.1, 1, 10]}, cv=3).fit(Xtr, y[tr])
        deltas.append(accuracy_score(y[te], rbf.predict(Xte)) - accuracy_score(y[te], lin.predict(Xte)))
    return np.mean(deltas), np.std(deltas)


for name, (X, y), pk in [("churn375", load_churn_sub(), 2), ("glass_pca2", load_glass(), 2), ("glass_full", load_glass(), 0)]:
    print(f"--- {name}: tuned RBF - linear ---")
    for seed in [42, 0, 7]:
        m, s = eval_pair(X, y, seed, pca_k=pk)
        flag = "ALIVE" if (m >= 0.02 and m > s) else "not-alive"
        print(f"seed={seed:2d} delta={m:+.4f} sigma_fold={s:.4f} -> {flag}")
