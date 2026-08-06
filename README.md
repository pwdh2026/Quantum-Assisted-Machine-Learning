# ⚛️ Classical vs Quantum SVM (PennyLane)

An interactive **Streamlit web application** for comparing **Classical Support Vector Machines (RBF Kernel)** and **Quantum Support Vector Machines (PennyLane-based Quantum Kernel)** under both traditional train-test evaluation and research-oriented cross-validation workflows.

The project provides an interactive environment for exploring the strengths, limitations, computational costs, and performance characteristics of classical and quantum kernel methods across multiple datasets.

---

## 🚀 Features

### 🧠 Classical vs Quantum Model Comparison

#### Classical SVM (RBF Kernel)

Uses the Radial Basis Function kernel:

```text
K(xi, xj) = exp(-γ ||xi - xj||²)
```

#### Quantum SVM

Uses PennyLane's `default.qubit` simulator to construct a quantum feature map and quantum kernel:

```text
K(x, y) = |<ψ(x)|ψ(y)>|²
```

Data is encoded into quantum states using parameterized rotations and entangling operations.

---

## 📊 Evaluation Modes

### Train-Test Mode

Traditional machine learning evaluation using a configurable train-test split.

Features:

* Adjustable test size fraction
* Classical vs Quantum comparison
* Metric variation across multiple sample splits
* Decision boundaries and confusion matrices
* Quantum kernel visualization

---

### 5-Fold Cross Validation Mode

Research-oriented evaluation using Stratified 5-Fold Cross Validation.

Features:

* Fold-wise model evaluation
* Average Accuracy, Precision, Recall, F1 Score, and Runtime
* Fold variation analysis
* Fold-safe preprocessing to prevent data leakage
* Fold-based visualizations and benchmarking

---

## 📂 Built-In Datasets

The application includes the following datasets:

* Iris
* Glass
* Social Network Ads
* Breast Cancer
* Wine

Users can also upload custom CSV datasets.

---

## 🧮 Preprocessing Pipeline

### Automatic Feature Handling

* Label Encoding
* One-Hot Encoding
* StandardScaler normalization

### Dimensionality Reduction

Optional PCA support for:

* Visualization
* Quantum kernel stability
* Reduced quantum feature space dimensionality

---

## ⚙️ Adjustable Parameters

| Parameter                 | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| Evaluation Mode           | Train-Test Split or 5-Fold CV                              |
| PCA Components            | Number of PCA dimensions used                              |
| Test Size Fraction        | Percentage of data used for testing (Train-Test Mode only) |
| Max Quantum Samples       | Limits quantum kernel computation size                     |
| Quantum Limitation Factor | Simulates practical quantum hardware constraints           |
| Dataset Selection         | Built-in datasets or custom CSV                            |

---

## 📈 Visualizations

### Performance Analysis

* Accuracy
* Precision
* Recall
* F1 Score
* Processing Time

### Model Diagnostics

* Classical Confusion Matrix
* Quantum Confusion Matrix
* Classical Decision Boundary
* Quantum Kernel Heatmap
* Sample Prediction Comparison

### Comparative Analysis

* Classical vs Quantum Metric Comparison
* Metric Variation Across Sample Splits
* Metric Variation Across Cross-Validation Folds
* Cross-Dataset Benchmark Summary

### Benchmark Visualizations

* Trend Plots
* Performance Comparison Charts
* Dataset Size Analysis
* Classical vs Quantum Runtime Comparison

---

## 🔬 Research-Oriented Features

The application supports:

* 5-Fold Cross Validation
* Fold-safe preprocessing
* Quantum resource limitation simulation
* Runtime benchmarking
* Multi-dataset evaluation
* Reproducible experimental settings

These additions make the project suitable for comparative Quantum Machine Learning studies and publication-oriented experimentation.

---

## 🧰 Tech Stack

| Component       | Technology          |
| --------------- | ------------------- |
| Frontend/UI     | Streamlit           |
| Classical ML    | Scikit-learn        |
| Quantum ML      | PennyLane           |
| Visualization   | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy       |

---

## 📂 Project Structure

```text
📦 classical-vs-quantum-svm
│
├── app.py
├── datasets/
│   ├── iris.csv
│   ├── glass.csv
│   ├── social_network_ads.csv
│   ├── breast_cancer.csv
│   ├── wine.csv
│   └── dataset_summary_results.csv
│
├── benchmark_results.csv
├── README.md
```

---

## 📚 Conceptual Overview

This project explores:

* Classical kernel-based classification
* Quantum-enhanced kernel methods
* Practical limitations of current quantum approaches
* Trade-offs between accuracy and computational cost
* Effects of dataset size and feature dimensionality on quantum models

The platform provides a side-by-side environment for evaluating whether quantum kernels offer meaningful benefits for specific classification problems.

---

## 🧠 Research Relevance

This work was inspired by and extends concepts explored in:

**Comparative Analysis of a Quantum SVM With an Optimized Kernel Versus Classical SVMs**

The project serves as a practical Quantum-Assisted Machine Learning (QAML) benchmarking environment and supports publication-oriented experimentation through reproducible evaluation workflows.

---

## 📜 License

Released under the MIT License.

---

## 👨‍💻 Author

**PlatinumManX**

🎓 Engineering Student
💻 Technical Game Development Enthusiast
⚛️ Quantum Machine Learning Explorer

GitHub: https://github.com/PlatinumManX

---

## 🔁 Reproducibility (GOAI AI for Research submission)

- Forked from: `PlatinumManX/Quantum-Assisted-Machine-Learning` @ commit `329688bbded76673010b0dff4aac099d34a5c9c4` (2026-06-10)
- Frozen environment: Python 3.13.6 / PennyLane 0.45.1 / scikit-learn 1.9.0 / Streamlit 1.61.1
- Entry points: `streamlit run app.py` (interactive comparison) and the parameterized scripts `d1_minimal_run.py`, `d2_leak_pair.py`, `d3_dose.py`, `d4_defective_seeds.py`
- Run log: `run_log.csv` records every configuration (dataset, PCA/qubits, protocol mode, q_limit, classical budget, seeds, metrics, times); negative and anomalous rows are retained
- Data: built-in UCI datasets ship with the repo; Bank Customer Churn must be obtained from Kaggle under its own license (not redistributed here)
