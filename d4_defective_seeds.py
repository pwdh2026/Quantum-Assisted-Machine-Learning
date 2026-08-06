"""d4: defective train-test protocol on seeds {42,0,7} (Bank Churn).

Mirrors app.py train-test semantics: scaler+PCA fit on ALL data before split
(leakage); classical trained on FULL train (untuned rbf + tuned 4-kernel);
quantum trained/tested on truncated prefix (300).
Seed 42 should reproduce O4 (Q=0.8367, C_untuned=0.7958).
Pairs with O5 matched+tuned per seed -> M signal (protocol factor) at 3 seeds.
"""
import time

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CSV = r"C:\Users\38277\.qwenworkcn\workspace\msg779bf8k5wk0rc\Quantum-Assisted-Machine-Learning\datasets\bank_churn.csv"
df = pd.read_csv(CSV)
X_df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
X_df = pd.get_dummies(X_df, columns=["Geography", "Gender"], drop_first=True)
y = df["Exited"].values
X_raw = X_df.values.astype(float)

PCA_K = 2
dev = qml.device("default.qubit", wires=PCA_K)


def fm(x):
    for i in range(len(x)):
        qml.RX(float(x[i]), wires=i)
        qml.RZ(float(x[i]) ** 2, wires=i)
    for i in range(len(x) - 1):
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev)
def circ(x):
    fm(x)
    return qml.state()


def K(A, B):
    sb = [circ(b) for b in B]
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        sa = circ(a)
        for j, s in enumerate(sb):
            out[i, j] = np.abs(np.vdot(sa, s)) ** 2
    return out


for seed in [42, 0, 7]:
    # leakage: fit on ALL data before split (app train-test semantics)
    sc = StandardScaler().fit(X_raw)
    p = PCA(PCA_K).fit(sc.transform(X_raw))
    Xp = p.transform(sc.transform(X_raw))
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xp, y, test_size=0.2, random_state=seed, stratify=y)

    cu = SVC(kernel="rbf").fit(X_tr, y_tr)
    acc_cu = accuracy_score(y_te, cu.predict(X_te))
    best_t, acc_ct = None, -1
    for kern in ["rbf", "linear", "poly", "sigmoid"]:
        m = GridSearchCV(SVC(kernel=kern), {"C": [0.1, 1, 10]}, cv=3).fit(X_tr, y_tr)
        a = accuracy_score(y_te, m.predict(X_te))
        if a > acc_ct:
            acc_ct, best_t = a, kern

    q_tr, y_q_tr = X_tr[:300], y_tr[:300]
    q_te, y_q_te = X_te[:300], y_te[:300]
    t0 = time.time()
    q = SVC(kernel="precomputed").fit(K(q_tr, q_tr), y_q_tr)
    acc_q = accuracy_score(y_q_te, q.predict(K(q_te, q_tr)))
    tq = time.time() - t0
    print(f"seed={seed:2d} | C_untuned={acc_cu:.4f} C_tuned={acc_ct:.4f}({best_t}) | "
          f"Q={acc_q:.4f} | Q-Cu={acc_q-acc_cu:+.4f} Q-Ct={acc_q-acc_ct:+.4f} | q_time={tq:.1f}s")
