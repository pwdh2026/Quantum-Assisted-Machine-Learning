"""D1 minimal run: QSVM (quantum kernel) vs classical SVMs on Iris.

Task slice: Iris-setosa vs Iris-versicolor, 2 features -> 2 qubits.
Protocol notes:
- Train/test split fixed with random_state=42 (recorded).
- StandardScaler fitted on TRAIN ONLY (fold-safe, no leakage).
- Quantum kernel K(x,y) = |<00|U^dag(y) U(x)|00>|^2 on default.qubit (CPU).
"""
import time

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

DATA = r"C:\Users\38277\.qwenworkcn\workspace\msg779bf8k5wk0rc\Quantum-Assisted-Machine-Learning\datasets\Iris.csv"

df = pd.read_csv(DATA)
df = df[df["Species"].isin(["Iris-setosa", "Iris-versicolor"])].copy()
X = df[["SepalLengthCm", "PetalLengthCm"]].values
y = LabelEncoder().fit_transform(df["Species"].values)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

n_qubits = X.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)


def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)
    qml.CNOT(wires=[0, 1])
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


@qml.qnode(dev)
def kernel_circuit(x, y_vec):
    feature_map(x)
    qml.adjoint(feature_map)(y_vec)
    return qml.probs(wires=range(n_qubits))


def quantum_kernel(A, B):
    K = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            K[i, j] = kernel_circuit(A[i], B[j])[0]
    return K


print("=== D1 minimal run: QSVM vs classical SVM (Iris binary, 2 qubits) ===")
print(f"pennylane={qml.version()}  samples={len(y)}  train={len(y_tr)}  test={len(y_te)}")
print(f"features={n_qubits} -> qubits={n_qubits}  device=default.qubit  split_seed=42")

t0 = time.time()
K_tr = quantum_kernel(X_tr, X_tr)
K_te = quantum_kernel(X_te, X_tr)
t_kernel = time.time() - t0

t0 = time.time()
qsvm = SVC(kernel="precomputed")
qsvm.fit(K_tr, y_tr)
acc_q = accuracy_score(y_te, qsvm.predict(K_te))
t_qsvm = time.time() - t0

t0 = time.time()
rbf = SVC(kernel="rbf").fit(X_tr, y_tr)
lin = SVC(kernel="linear").fit(X_tr, y_tr)
t_classical = time.time() - t0
acc_rbf = accuracy_score(y_te, rbf.predict(X_te))
acc_lin = accuracy_score(y_te, lin.predict(X_te))

print("--- results (test accuracy) ---")
print(f"classical SVM (RBF)     : {acc_rbf:.4f}   fit_time={t_classical:.3f}s (both classical models)")
print(f"classical SVM (linear)  : {acc_lin:.4f}")
print(f"quantum kernel SVM      : {acc_q:.4f}   kernel_eval_time={t_kernel:.1f}s  fit_time={t_qsvm:.3f}s")
print("--- D1 verdict ---")
print("Environment OK: PennyLane default.qubit simulation + sklearn baselines ran end-to-end.")
