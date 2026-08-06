"""D2 paired living-example: defective vs fixed protocol on identical data/seed.

Defective protocol (mirrors app.py train-test mode, L126-135 + L280-287):
  - StandardScaler + PCA fitted on ALL data BEFORE the train/test split (leakage)
  - classical SVM trained on the FULL training split
  - quantum SVM trained on a TRUNCATED prefix of the training split (q limit),
    evaluated on (possibly truncated) test
Fixed protocol (fold-safe):
  - scaler + PCA fitted on the training split ONLY
  - classical and quantum trained/evaluated on the SAME splits (no asymmetry)

Same feature map as app.py (RX(x_i), RZ(x_i^2), linear CNOT chain), same seed.
"""
import time

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

BASE = r"C:\Users\38277\.qwenworkcn\workspace\msg779bf8k5wk0rc\Quantum-Assisted-Machine-Learning\datasets"
PCA_K = 2
Q_LIMIT = 100
SEED = 42


def load(name):
    df = pd.read_csv(f"{BASE}/{name}")
    X = df.drop(columns=[df.columns[-1]]).values.astype(float)
    y = LabelEncoder().fit_transform(df[df.columns[-1]].astype(str))
    return X, y


def make_kernel(n_wires):
    dev = qml.device("default.qubit", wires=n_wires)

    def feature_map(x):
        for i in range(len(x)):
            qml.RX(float(x[i]), wires=i)
            qml.RZ(float(x[i]) ** 2, wires=i)
        for i in range(len(x) - 1):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev)
    def circuit(x):
        feature_map(x)
        return qml.state()

    def K(A, B):
        states_b = [circuit(b) for b in B]
        out = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            sa = circuit(a)
            for j, sb in enumerate(states_b):
                out[i, j] = np.abs(np.vdot(sa, sb)) ** 2
        return out

    return K


def run_one(X, y, leak: bool):
    if leak:
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        p = PCA(n_components=PCA_K).fit(Xs)
        Xp = p.transform(Xs)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xp, y, test_size=0.2, random_state=SEED, stratify=y)
    else:
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
        sc = StandardScaler().fit(X_tr_raw)
        p = PCA(n_components=PCA_K).fit(sc.transform(X_tr_raw))
        X_tr = p.transform(sc.transform(X_tr_raw))
        X_te = p.transform(sc.transform(X_te_raw))

    clf = SVC(kernel="rbf").fit(X_tr, y_tr)
    acc_c = accuracy_score(y_te, clf.predict(X_te))

    K = make_kernel(PCA_K)
    q_tr = X_tr[:Q_LIMIT]
    y_q_tr = y_tr[:Q_LIMIT]
    q_te = X_te[:Q_LIMIT]
    y_q_te = y_te[:Q_LIMIT]
    t0 = time.time()
    Ktr = K(q_tr, q_tr)
    Kte = K(q_te, q_tr)
    qclf = SVC(kernel="precomputed").fit(Ktr, y_q_tr)
    acc_q = accuracy_score(y_q_te, qclf.predict(Kte))
    tq = time.time() - t0
    return acc_c, acc_q, tq


for name in ["wine.csv", "glass.csv"]:
    X, y = load(name)
    acc_c_d, acc_q_d, t_d = run_one(X, y, leak=True)
    acc_c_f, acc_q_f, t_f = run_one(X, y, leak=False)
    print(f"=== {name} (PCA={PCA_K}, seed={SEED}, q_limit={Q_LIMIT}) ===")
    print(f"defective: acc_c={acc_c_d:.4f} acc_q={acc_q_d:.4f} delta={acc_q_d-acc_c_d:+.4f} q_time={t_d:.1f}s")
    print(f"fixed    : acc_c={acc_c_f:.4f} acc_q={acc_q_f:.4f} delta={acc_q_f-acc_c_f:+.4f} q_time={t_f:.1f}s")
    print(f"protocol effect on delta: {(acc_q_d-acc_c_d)-(acc_q_f-acc_c_f):+.4f}")
