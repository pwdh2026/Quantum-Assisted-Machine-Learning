import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================= Datasets =================
datasets = {
    "Iris": "datasets/iris.csv",
    "Glass": "datasets/glass.csv",
    "Social Media Ads": "datasets/social_network_ads.csv",
    "Breast Cancer": "datasets/breast_cancer.csv",
    "Wine": "datasets/wine.csv"
}

# ================= CSV file =================
os.makedirs("datasets", exist_ok=True)
results_file = "datasets/dataset_summary_results.csv"

results = []

# ================= Strided Sampling =================
def get_fraction_rows(X, y, fraction):
    n = len(X)
    
    if fraction == 0.3:
        # Take 1, skip 2
        idx = np.arange(0, n, 3)
    elif fraction == 0.6:
        # Take 2, skip 1
        idx = np.concatenate([np.arange(i, i+2) for i in range(0, n, 3)])
        idx = idx[idx < n]
    elif fraction == 1.0:
        idx = np.arange(n)
    else:
        idx = np.arange(int(n*fraction))
    
    return X[idx], y[idx]

# ================= Run Experiments =================
fractions = [0.3, 0.6, 1.0]

for ds_name, path in datasets.items():
    df = pd.read_csv(path)
    target_col = df.columns[-1]

    X_df = df.drop(columns=[target_col])
    y_ser = df[target_col]

    # Handle categorical features
    categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_ser.astype(str))
    X = X_df.values.astype(float)

    for frac in fractions:
        X_subset, y_subset = get_fraction_rows(X, y, frac)

        # Skip if only one class present
        if len(np.unique(y_subset)) < 2:
            print(f"Skipping {ds_name} at {int(frac*100)}%: only 1 class present")
            continue

        # Train/test split
        stratify_val = y_subset if len(np.unique(y_subset)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42, stratify=stratify_val
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Classical SVM
        start_time = time.time()
        clf = SVC(kernel="rbf")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - start_time

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Dataset": ds_name,
            "Data Used": f"{int(frac*100)}%",
            "Classical Accuracy": acc,
            "Quantum Accuracy": "",
            "Classical Precision": prec,
            "Quantum Precision": "",
            "Classical Recall": rec,
            "Quantum Recall": "",
            "Classical F1 Score": f1,
            "Quantum F1 Score": "",
            "Classical Time (s)": round(elapsed, 4),
            "Quantum Time (s)": ""
        })

# ================= Save CSV =================
columns = [
    "Dataset", "Data Used",
    "Classical Accuracy", "Quantum Accuracy",
    "Classical Precision", "Quantum Precision",
    "Classical Recall", "Quantum Recall",
    "Classical F1 Score", "Quantum F1 Score",
    "Classical Time (s)", "Quantum Time (s)"
]

df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(results_file, index=False)
print(f"Results saved to {results_file}")
