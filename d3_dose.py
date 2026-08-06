"""O5: Bank Churn dose-response + tuned-classical survival test.

Protocol (fair): both models see the SAME subsample; per-fold fold-safe
scaler+PCA(2); Stratified 5-fold CV; seeds {42,0,7}.
Doses mirror app CV semantics: total_limit = min(n, int(q_limit/0.8)),
subsample permutation fixed at RandomState(42) (app behavior).
Quantum = app feature map RX(x_i),RZ(x_i^2),CNOT chain; state-cached kernel.
Classical budgets: untuned {rbf,linear,poly,sigmoid} + grid C in {.1,1,10}.
Repro checks (seed 42): O4 CV quantum ~0.8027 (375 subsample),
O4 CV classical full-data untuned rbf ~0.7980.
"""
import time

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

T0 = time.time()
CSV = r"C:\Users\38277\.qwenworkcn\workspace\msg779bf8k5wk0rc\Quantum-Assisted-Machine-Learning\datasets\bank_churn.csv"
df = pd.read_csv(CSV)
X_df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
X_df = pd.get_dummies(X_df, columns=["Geography", "Gender"], drop_first=True)
y = df["Exited"].values
X_raw = X_df.values.astype(float)
print(f"rows={len(df)} features={X_raw.shape[1]} exited_rate={y.mean():.4f} majority_baseline={1-y.mean():.4f}")

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


def prep(tr, te, X):
    sc = StandardScaler().fit(X[tr])
    p = PCA(PCA_K).fit(sc.transform(X[tr]))
    return p.transform(sc.transform(X[tr])), p.transform(sc.transform(X[te]))


def cv_quantum(Xs, ys, seed):
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
        Xtr, Xte = prep(tr, te, Xs)
        q = SVC(kernel="precomputed").fit(K(Xtr, Xtr), ys[tr])
        accs.append(accuracy_score(ys[te], q.predict(K(Xte, Xtr))))
    return np.mean(accs), np.std(accs)


def cv_classical(Xs, ys, seed, budget):
    res = {}
    for kern in ["rbf", "linear", "poly", "sigmoid"]:
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys):
            Xtr, Xte = prep(tr, te, Xs)
            if budget == "untuned":
                m = SVC(kernel=kern).fit(Xtr, ys[tr])
            else:
                m = GridSearchCV(SVC(kernel=kern), {"C": [0.1, 1, 10]}, cv=3).fit(Xtr, ys[tr])
            accs.append(accuracy_score(ys[te], m.predict(Xte)))
        res[kern] = np.mean(accs)
    return res


# ---- repro O4 classical full-data (seed 42, untuned rbf)
r = cv_classical(X_raw, y, 42, "untuned")
print(f"REPRO classical full-data untuned: rbf={r['rbf']:.4f} (O4 said 0.7980)")

for dose in [300, 150, 75]:
    total = min(len(X_raw), int(dose / 0.8))
    idx = np.random.RandomState(42).permutation(len(X_raw))[:total]
    Xs, ys = X_raw[idx], y[idx]
    for seed in [42, 0, 7]:
        aq, sq = cv_quantum(Xs, ys, seed)
        cu = cv_classical(Xs, ys, seed, "untuned")
        cg = cv_classical(Xs, ys, seed, "grid")
        bu, bg = max(cu, key=cu.get), max(cg, key=cg.get)
        tag = "  <-- repro O4 Q" if (dose == 300 and seed == 42) else ""
        print(f"dose={dose:3d} seed={seed:2d} n={total:3d} | Q={aq:.4f}+-{sq:.4f} | "
              f"untuned_best={cu[bu]:.4f}({bu}) | tuned_best={cg[bg]:.4f}({bg}) | "
              f"Q-tuned={aq-cg[bg]:+.4f}{tag}")
        if dose == 300 and seed == 42:
            print(f"   REPRO check: O4 CV quantum said 0.8027")

print(f"elapsed={time.time()-T0:.0f}s")
