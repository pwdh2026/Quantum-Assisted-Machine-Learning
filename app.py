import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Classical vs Quantum SVM (PennyLane)")

st.markdown("""
    <h1 style="display:flex; align-items:center; gap:10px;">
        ⚛️ Classical vs Quantum SVM (PennyLane)
    </h1>
""", unsafe_allow_html=True)

# ================= Dataset Selection =================
st.sidebar.header("Dataset Selection")
dataset_options = {
    "Upload CSV": None,
    "Iris": "datasets/iris.csv",
    "Glass": "datasets/glass.csv",
    "Social Media Ads": "datasets/social_network_ads.csv",
    "Breast Cancer": "datasets/breast_cancer.csv",
    "Wine": "datasets/wine.csv"
}

dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    ["Upload CSV", "Iris", "Glass", "Social Media Ads", "Breast Cancer", "Wine"],
    index=0
)

df = None
uploaded_file = None

if dataset_choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV dataset (last column may be label or choose label)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(dataset_options[dataset_choice])

# ================= Sidebar Parameters =================
use_cv_mode = st.sidebar.toggle("🔄 Use 5-Fold CV Mode", value=False)
cv_mode = "5-Fold CV Mode" if use_cv_mode else "Train-Test Mode"

pca_components = st.sidebar.slider("PCA components for Quantum kernel (0 = no PCA)", 0, 5, 2, 1)

if cv_mode == "Train-Test Mode":
    test_size = st.sidebar.slider("Test size fraction", 0.1, 0.5, 0.2, 0.05)
else:
    test_size = 0.2

quantum_samples_limit = st.sidebar.number_input("Max training samples for quantum kernel", min_value=10, max_value=300, value=100, step=10)

quantum_limit_factor = st.sidebar.slider(
    "Quantum Limitations (fraction of training samples used)",
    0.1, 1.0, 1.0, 0.1,
    help="Simulate hardware constraints by reducing the training set size"
)

# ================= Technical Information Panel =================
with st.sidebar.expander("ℹ️ Technical Information", expanded=False):
    st.markdown("""
    ### Dataset Handling
    - Upload custom CSV or use sample datasets (Iris, Glass, Social Media Ads).
    - Target column is selected (default: last column).
    - Categorical features are one-hot encoded; labels are encoded with `LabelEncoder`.

    ### Classical SVM
    - Kernel: Radial Basis Function (RBF).
    - Fits the data in feature space using:
      $$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

    ### Quantum SVM
    - Uses PennyLane simulation backend (`default.qubit`).
    - Data encoded via parameterized rotations (feature map).
    - Quantum kernel built from inner product of quantum states:
      $$K(x, y) = |\langle \psi(x) | \psi(y) \rangle|^2$$

    ### Evaluation
    - Accuracy, Precision, Recall, and F1-Score are reported.
    - If 2D (after PCA), decision boundary and scatter plots are shown.
    - Quantum kernel matrix is visualized with a heatmap.
    """)

# ================= Preprocessing =================
if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.head())

    cols = list(df.columns)
    target = None

    if dataset_choice == "Upload CSV":
        target = st.selectbox("Select target column", options=['(last column)'] + cols)
        if target == "(last column)":
            target = cols[-1]
    else:
        target = cols[-1]

    if target is not None:
        X_df = df.drop(columns=[target])
        y_ser = df[target]

        categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)

        le_target = LabelEncoder()
        y = le_target.fit_transform(y_ser.astype(str))

        X_raw = X_df.values.astype(float)

        unique, counts = np.unique(y, return_counts=True)
        stratify_val = y if counts.min() >= 2 else None

        if cv_mode == "Train-Test Mode":
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            if pca_components > 0 and X.shape[1] > pca_components:
                pca = PCA(n_components=pca_components)
                X = pca.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=42, stratify=stratify_val
            )

        # ================= Classical SVM =================
        st.markdown("## Classical SVM")

        if cv_mode == "Train-Test Mode":
            start_time = time.time()
            clf = SVC(kernel="rbf", probability=False)
            clf.fit(X_train, y_train)
            y_pred_c = clf.predict(X_test)
            classical_time = time.time() - start_time

            acc_c = accuracy_score(y_test, y_pred_c)
            prec_c = precision_score(y_test, y_pred_c, average='weighted', zero_division=0)
            rec_c = recall_score(y_test, y_pred_c, average='weighted', zero_division=0)
            f1_c = f1_score(y_test, y_pred_c, average='weighted', zero_division=0)
            cm_c = confusion_matrix(y_test, y_pred_c)

            X_plot = X
            y_plot = y
            clf_plot = clf

        else: # 5-Fold CV Mode
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if stratify_val is not None else KFold(n_splits=5, shuffle=True, random_state=42)

            accs, precs, recs, f1s, times = [], [], [], [], []
            fold = 0

            for train_idx, test_idx in skf.split(X_raw, y):
                fold += 1
                X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_raw)
                X_te = scaler.transform(X_te_raw)

                if pca_components > 0 and X_tr.shape[1] > pca_components:
                    pca = PCA(n_components=pca_components)
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)

                start_time = time.time()
                clf = SVC(kernel="rbf", probability=False)
                clf.fit(X_tr, y_tr)
                y_pr = clf.predict(X_te)
                elapsed = time.time() - start_time

                accs.append(accuracy_score(y_te, y_pr))
                precs.append(precision_score(y_te, y_pr, average='weighted', zero_division=0))
                recs.append(recall_score(y_te, y_pr, average='weighted', zero_division=0))
                f1s.append(f1_score(y_te, y_pr, average='weighted', zero_division=0))
                times.append(elapsed)

                if fold == 1:
                    y_pred_c = y_pr
                    X_test = X_te
                    y_test = y_te
                    cm_c = confusion_matrix(y_te, y_pr)
                    X_plot = np.vstack((X_tr, X_te))
                    y_plot = np.concatenate((y_tr, y_te))
                    clf_plot = clf

            acc_c = np.mean(accs)
            prec_c = np.mean(precs)
            rec_c = np.mean(recs)
            f1_c = np.mean(f1s)
            classical_time = np.mean(times)
            classical_cv_metrics = {"accs": accs, "precs": precs, "recs": recs, "f1s": f1s}

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{acc_c*100:.2f}%")
        col2.metric("Precision", f"{prec_c*100:.2f}%")
        col3.metric("Recall", f"{rec_c*100:.2f}%")
        col4.metric("F1 Score", f"{f1_c*100:.2f}%")
        col5.metric("Processing Time", f"{classical_time:.2f} s")

        # Classical Confusion Matrix
        st.markdown("<h5 style='font-size:18px;'>Classical Confusion Matrix</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3, 2.5))
        sns.heatmap(pd.DataFrame(cm_c), annot=True, cmap="YlGnBu", fmt="d", ax=ax, cbar=False)
        plt.tight_layout(pad=0.2)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Classical Confusion Matrix")
        st.pyplot(fig, use_container_width=False)

        if (cv_mode == "Train-Test Mode" and X.shape[1] == 2) or (cv_mode == "5-Fold CV Mode" and X_plot.shape[1] == 2):
            st.subheader("Classical Decision Boundary")
            h = .02
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf_plot.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots()
            contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
            ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap="coolwarm", edgecolor='k')

            unique_classes = np.unique(y_plot)
            class_names = le_target.inverse_transform(unique_classes)
            for cls, name in zip(unique_classes, class_names):
                mask = (Z == cls)
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0:
                    x_center = np.mean(xx[y_coords, x_coords])
                    y_center = np.mean(yy[y_coords, x_coords])
                    ax.text(x_center, y_center, str(name),
                            color="black", fontsize=10, fontweight='bold',
                            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.set_title("Classical Decision Boundary with Class Labels")
            st.pyplot(fig)

        # ================= Quantum SVM =================
        st.markdown("## Quantum SVM (PennyLane kernel)")

        n_features = pca_components if pca_components > 0 else X_raw.shape[1]
        n_wires = n_features if n_features > 0 else 1
        dev = qml.device("default.qubit", wires=n_wires)

        def feature_map(x):
            for i in range(len(x)):
                qml.RX(float(x[i]), wires=i)
                qml.RZ(float(x[i])**2, wires=i)
            for i in range(len(x)-1):
                qml.CNOT(wires=[i, i+1])

        @qml.qnode(dev)
        def kernel_circuit(x):
            feature_map(x)
            return qml.state()

        def quantum_kernel_matrix(X1, X2):
            m, n = len(X1), len(X2)
            K = np.zeros((m, n))
            states_X2 = [kernel_circuit(pnp.array(x2)) for x2 in X2]
            for i in range(m):
                state1 = kernel_circuit(pnp.array(X1[i]))
                for j in range(n):
                    K[i, j] = np.abs(np.vdot(state1, states_X2[j]))**2
            return K

        if cv_mode == "Train-Test Mode":
            effective_train_size = int(len(X_train) * quantum_limit_factor)
            effective_train_size = min(effective_train_size, quantum_samples_limit)

            X_train_q = X_train[:effective_train_size]
            y_train_q = y_train[:effective_train_size]

            X_test_q = X_test if X_test.shape[0] <= quantum_samples_limit else X_test[:quantum_samples_limit]
            y_test_q = y_test if y_test.shape[0] <= quantum_samples_limit else y_test[:quantum_samples_limit]

            st.caption(f"⚠️ Quantum limitation applied: Using {effective_train_size} training samples "
                       f"({quantum_limit_factor*100:.0f}% of available training data).")

            try:
                start_time = time.time()
                K_train = quantum_kernel_matrix(X_train_q, X_train_q)
                K_test = quantum_kernel_matrix(X_test_q, X_train_q)
                qclf = SVC(kernel="precomputed")
                qclf.fit(K_train, y_train_q)
                y_pred_q = qclf.predict(K_test)
                quantum_time = time.time() - start_time

                acc_q = accuracy_score(y_test_q, y_pred_q)
                prec_q = precision_score(y_test_q, y_pred_q, average='weighted', zero_division=0)
                rec_q = recall_score(y_test_q, y_pred_q, average='weighted', zero_division=0)
                f1_q = f1_score(y_test_q, y_pred_q, average='weighted', zero_division=0)
                cm_q = confusion_matrix(y_test_q, y_pred_q)

            except Exception as e:
                st.error("Quantum kernel computation failed: " + str(e))
                acc_q = prec_q = rec_q = f1_q = quantum_time = None
                y_pred_q = None

        else: # 5-Fold CV Mode
            total_limit = min(int(len(X_raw) * quantum_limit_factor), int(quantum_samples_limit / 0.8))

            indices = np.random.RandomState(42).permutation(len(X_raw))[:total_limit]
            X_q_raw = X_raw[indices]
            y_q_full = y[indices]

            st.caption(f"⚠️ Quantum limitation applied BEFORE CV: Subsampled dataset to {total_limit} total samples "
                       f"to ensure average training samples per fold <= {quantum_samples_limit}.")

            skf_q = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if len(np.unique(y_q_full)) > 1 else KFold(n_splits=5, shuffle=True, random_state=42)

            accs_q, precs_q, recs_q, f1s_q, times_q = [], [], [], [], []
            fold_q = 0
            y_pred_q = None

            try:
                for train_idx, test_idx in skf_q.split(X_q_raw, y_q_full):
                    fold_q += 1
                    X_tr_raw, X_te_raw = X_q_raw[train_idx], X_q_raw[test_idx]
                    y_tr, y_te = y_q_full[train_idx], y_q_full[test_idx]

                    scaler_q = StandardScaler()
                    X_tr = scaler_q.fit_transform(X_tr_raw)
                    X_te = scaler_q.transform(X_te_raw)

                    if pca_components > 0 and X_tr.shape[1] > pca_components:
                        pca_q = PCA(n_components=pca_components)
                        X_tr = pca_q.fit_transform(X_tr)
                        X_te = pca_q.transform(X_te)

                    start_time = time.time()
                    K_train = quantum_kernel_matrix(X_tr, X_tr)
                    K_test = quantum_kernel_matrix(X_te, X_tr)
                    qclf = SVC(kernel="precomputed")
                    qclf.fit(K_train, y_tr)
                    y_pr = qclf.predict(K_test)
                    elapsed = time.time() - start_time

                    accs_q.append(accuracy_score(y_te, y_pr))
                    precs_q.append(precision_score(y_te, y_pr, average='weighted', zero_division=0))
                    recs_q.append(recall_score(y_te, y_pr, average='weighted', zero_division=0))
                    f1s_q.append(f1_score(y_te, y_pr, average='weighted', zero_division=0))
                    times_q.append(elapsed)

                    if fold_q == 1:
                        X_test_q = X_te
                        y_test_q = y_te
                        y_pred_q = y_pr
                        cm_q = confusion_matrix(y_te, y_pr)
                        K_train_plot = K_train

                acc_q = np.mean(accs_q)
                prec_q = np.mean(precs_q)
                rec_q = np.mean(recs_q)
                f1_q = np.mean(f1s_q)
                quantum_time = np.mean(times_q)
                quantum_cv_metrics = {"accs": accs_q, "precs": precs_q, "recs": recs_q, "f1s": f1s_q}
                K_train = K_train_plot 

            except Exception as e:
                st.error("Quantum kernel computation failed: " + str(e))
                acc_q = prec_q = rec_q = f1_q = quantum_time = None
                y_pred_q = None

        if acc_q is not None:
            col6, col7, col8, col9, col10 = st.columns(5)
            col6.metric("Accuracy", f"{acc_q*100:.2f}%")
            col7.metric("Precision", f"{prec_q*100:.2f}%")
            col8.metric("Recall", f"{rec_q*100:.2f}%")
            col9.metric("F1 Score", f"{f1_q*100:.2f}%")
            col10.metric("Processing Time", f"{quantum_time:.2f} s")

            # Quantum Confusion Matrix
            st.markdown("<h5 style='font-size:18px;'>Quantum Confusion Matrix</h5>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sns.heatmap(pd.DataFrame(cm_q), annot=True, cmap="YlGnBu", fmt="d", ax=ax, cbar=False)
            plt.tight_layout(pad=0.2)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Quantum Confusion Matrix")
            st.pyplot(fig, use_container_width=False)

            st.subheader("Quantum Kernel Matrix Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(K_train, cmap="viridis", ax=ax)
            st.pyplot(fig)

        # ================= Metric Comparison =================
        st.markdown("## Metric Comparison")

        st.markdown("""
            <div style="text-align:center; background-color:#f0f2f6; padding:10px; border-radius:5px; font-size:30px; font-weight:bold;">
                🟦 <span style="color:steelblue;">Classical SVM</span> &nbsp;&nbsp;&nbsp; 
                🟧 <span style="color:orange;">Quantum SVM</span>
            </div>
        """, unsafe_allow_html=True)

        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        classical_values = [acc_c, prec_c, rec_c, f1_c]
        quantum_values = [acc_q, prec_q, rec_q, f1_q]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(4, 1.5))
        ax.bar(x - width/2, classical_values, width, color='steelblue')
        ax.bar(x + width/2, quantum_values, width, color='orange')

        ax.set_ylabel('Score')
        ax.set_title('Classical vs Quantum SVM Metric Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        st.pyplot(fig)

        # ================= Sample Predictions =================
        st.markdown("## Sample predictions comparison (first rows)")
        sample_n = min(10, len(y_test))
        df_comp = pd.DataFrame({"Actual": y_test[:sample_n], "Classical": y_pred_c[:sample_n]})
        if y_pred_q is not None:
            sample_n_q = min(sample_n, len(y_pred_q))
            df_comp["Quantum"] = pd.Series(y_pred_q[:sample_n_q])
        st.dataframe(df_comp)

        # ================= Metric Variation Across Dataset =================
        if cv_mode == "Train-Test Mode":
            st.markdown("## Metric Variation Across Dataset (10 Sample Splits)")

            n_splits = 10
            split_size = len(X_test) // n_splits
            variation_data = []

            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X_test)

                y_true_split = y_test[start_idx:end_idx]
                y_pred_c_split = y_pred_c[start_idx:end_idx]

                if y_pred_q is not None:
                    y_pred_q_split = y_pred_q[start_idx:end_idx if i < n_splits - 1 else len(y_test_q)]

                acc_c_split = accuracy_score(y_true_split, y_pred_c_split)
                prec_c_split = precision_score(y_true_split, y_pred_c_split, average='weighted', zero_division=0)
                rec_c_split = recall_score(y_true_split, y_pred_c_split, average='weighted', zero_division=0)
                f1_c_split = f1_score(y_true_split, y_pred_c_split, average='weighted', zero_division=0)

                if y_pred_q is not None:
                    acc_q_split = accuracy_score(y_test_q[start_idx:end_idx], y_pred_q_split)
                    prec_q_split = precision_score(y_test_q[start_idx:end_idx], y_pred_q_split, average='weighted', zero_division=0)
                    rec_q_split = recall_score(y_test_q[start_idx:end_idx], y_pred_q_split, average='weighted', zero_division=0)
                    f1_q_split = f1_score(y_test_q[start_idx:end_idx], y_pred_q_split, average='weighted', zero_division=0)
                else:
                    acc_q_split = prec_q_split = rec_q_split = f1_q_split = np.nan

                variation_data.append([
                    f"Split {i+1}",
                    acc_c_split, prec_c_split, rec_c_split, f1_c_split,
                    acc_q_split, prec_q_split, rec_q_split, f1_q_split
                ])

        else:
            st.markdown("## Metric Variation Across Folds")
            variation_data = []

            for i in range(5):
                acc_c_split = classical_cv_metrics["accs"][i]
                prec_c_split = classical_cv_metrics["precs"][i]
                rec_c_split = classical_cv_metrics["recs"][i]
                f1_c_split = classical_cv_metrics["f1s"][i]

                if 'quantum_cv_metrics' in locals() and quantum_cv_metrics is not None:
                    acc_q_split = quantum_cv_metrics["accs"][i]
                    prec_q_split = quantum_cv_metrics["precs"][i]
                    rec_q_split = quantum_cv_metrics["recs"][i]
                    f1_q_split = quantum_cv_metrics["f1s"][i]
                else:
                    acc_q_split = prec_q_split = rec_q_split = f1_q_split = np.nan

                variation_data.append([
                    f"Fold {i+1}",
                    acc_c_split, prec_c_split, rec_c_split, f1_c_split,
                    acc_q_split, prec_q_split, rec_q_split, f1_q_split
                ])

        columns = pd.MultiIndex.from_tuples([
            ("Split" if cv_mode == "Train-Test Mode" else "Fold", ""),
            ("Classical", "Accuracy"),
            ("Classical", "Precision"),
            ("Classical", "Recall"),
            ("Classical", "F1 Score"),
            ("Quantum", "Accuracy"),
            ("Quantum", "Precision"),
            ("Quantum", "Recall"),
            ("Quantum", "F1 Score")
        ])

        df_variation = pd.DataFrame(variation_data, columns=columns)

        styled_df = (
            df_variation.style
            .format("{:.2f}", subset=pd.IndexSlice[:, df_variation.columns.get_level_values(0) != ("Split" if cv_mode == "Train-Test Mode" else "Fold")])
            .set_table_styles([
                {'selector': 'th.col_heading.level0', 'props': [('text-align', 'center')]},
                {'selector': 'th.col_heading.level1', 'props': [('text-align', 'center')]},
            ])
            .background_gradient(axis=None, cmap="Blues")
        )
        st.dataframe(styled_df)

        # ================= Scatter Plots =================
        if (cv_mode == "Train-Test Mode" and X.shape[1] == 2) or (cv_mode == "5-Fold CV Mode" and X_plot.shape[1] == 2):
            st.subheader("Scatter Plots: Classical vs Quantum Predictions")
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                fig, ax = plt.subplots()
                correct_idx = y_test == y_pred_c
                wrong_idx = ~correct_idx
                ax.scatter(X_test[correct_idx, 0], X_test[correct_idx, 1],
                           c=y_pred_c[correct_idx], cmap="coolwarm", marker='o', edgecolor='k')
                ax.scatter(X_test[wrong_idx, 0], X_test[wrong_idx, 1],
                           c=y_pred_c[wrong_idx], cmap="coolwarm", marker='x', edgecolor='k')
                ax.set_title("Classical SVM Predictions (o=correct, x=wrong)")
                st.pyplot(fig)

            if y_pred_q is not None:
                with col_s2:
                    fig, ax = plt.subplots()
                    correct_idx = y_test_q == y_pred_q
                    wrong_idx = ~correct_idx
                    ax.scatter(X_test_q[correct_idx, 0], X_test_q[correct_idx, 1],
                               c=y_pred_q[correct_idx], cmap="coolwarm", marker='o', edgecolor='k')
                    ax.scatter(X_test_q[wrong_idx, 0], X_test_q[wrong_idx, 1],
                               c=y_pred_q[wrong_idx], cmap="coolwarm", marker='x', edgecolor='k')
                    ax.set_title("Quantum SVM Predictions (o=correct, x=wrong)")
                    st.pyplot(fig)

                if 'results_summary' not in st.session_state:
                    st.session_state.results_summary = {}

                if dataset_choice != "Upload CSV":
                    st.session_state.results_summary[dataset_choice] = {
                        "Classical Accuracy": acc_c,
                        "Classical Precision": prec_c,
                        "Classical Recall": rec_c,
                        "Classical F1": f1_c,
                        "Classical Time (s)": classical_time,
                        "Quantum Accuracy": acc_q,
                        "Quantum Precision": prec_q,
                        "Quantum Recall": rec_q,
                        "Quantum F1": f1_q,
                        "Quantum Time (s)": quantum_time,
                    }

# ================= Summary Toggle  =================
show_summary = st.sidebar.toggle("📋 Show Dataset Summary", value=False)

if show_summary:
    st.markdown("## 📊 Cross-Dataset Summary")

    try:
        summary_df = pd.read_csv("datasets/dataset_summary_results.csv")

        styled_summary = ( 
            summary_df.style
            .format({
                "Classical Accuracy": "{:.2%}",  
                "Classical Precision": "{:.2%}",
                "Classical Recall": "{:.2%}",
                "Classical F1 Score": "{:.2%}",
                "Classical Time (s)": lambda x: f"{x}s",   
                "Quantum Accuracy": "{:.2%}",  
                "Quantum Precision": "{:.2%}",
                "Quantum Recall": "{:.2%}",
                "Quantum F1 Score": "{:.2%}",
                "Quantum Time (s)": lambda x: f"{x}s"     
            })
            .background_gradient(axis=None, cmap="coolwarm")
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('text-align', 'center')]},
            ])
        )

        st.dataframe(styled_summary, use_container_width=True)

       # ================= Graph Comparison =================
        st.markdown("## 📈 Performance Comparison Across Dataset Sizes 🟦 Classical & 🟧 Quantum")

        summary_df['Data Used'] = summary_df['Data Used'].astype(str).str.replace('%', '').astype(float)
        summary_df = summary_df.sort_values(by=['Dataset', 'Data Used'])

        metric_choice = st.selectbox(
            "Select metric to visualize:",
            ["Accuracy", "Precision", "Recall", "F1 Score", "Time (s)"],
            key="metric_select_summary"
        )

        datasets = summary_df['Dataset'].unique()
        num_datasets = len(datasets)

        last_idx = num_datasets if num_datasets % 2 == 0 else num_datasets - 1

        col_index = 0
        cols = st.columns(2)

        for i in range(last_idx):
            dataset = datasets[i]
            subset = summary_df[summary_df['Dataset'] == dataset]

            fig, ax = plt.subplots(figsize=(3.2, 2.0))
            ax.plot(subset['Data Used'], subset[f'Classical {metric_choice}'], marker='o', label='Classical', color='steelblue')
            ax.plot(subset['Data Used'], subset[f'Quantum {metric_choice}'], marker='o', label='Quantum', color='orange')
            ax.set_title(f"{dataset}", fontsize=9)
            ax.set_xlabel("Size (%)", fontsize=7)
            ax.set_ylabel(metric_choice, fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout(pad=0.3)

            with cols[col_index]:
                st.pyplot(fig, use_container_width=False)

            col_index += 1
            if col_index == 2:
                cols = st.columns(2)
                col_index = 0
            plt.close(fig)

        if num_datasets % 2 != 0:
            last_dataset = datasets[-1]
            subset = summary_df[summary_df['Dataset'] == last_dataset]

            cols = st.columns([1, 2, 1])  
            fig, ax = plt.subplots(figsize=(3.2, 2.0))
            ax.plot(subset['Data Used'], subset[f'Classical {metric_choice}'], marker='o', label='Classical', color='steelblue')
            ax.plot(subset['Data Used'], subset[f'Quantum {metric_choice}'], marker='o', label='Quantum', color='orange')
            ax.set_title(f"{last_dataset}", fontsize=9)
            ax.set_xlabel("Size (%)", fontsize=7)
            ax.set_ylabel(metric_choice, fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout(pad=0.3)

            with cols[1]:  
                st.pyplot(fig, use_container_width=False)

            plt.close(fig)


        # ================= Bar Chart Comparison =================
        st.markdown("## 📊 Performance Comparison (Bar Charts) 🟦 Classical & 🟧 Quantum")

        metric_choice_bar = st.selectbox(
            "Select metric to visualize as Bar Chart:",
            ["Accuracy", "Precision", "Recall", "F1 Score", "Time (s)"],
            key="metric_select_summary_bar_chart"
        )

        summary_df = summary_df.sort_values(by=['Dataset', 'Data Used'])

        datasets = summary_df['Dataset'].unique()
        num_datasets = len(datasets)

        col_index = 0
        cols = st.columns(2)

        last_idx = num_datasets if num_datasets % 2 == 0 else num_datasets - 1

        for i in range(last_idx):
            dataset = datasets[i]
            subset = summary_df[summary_df['Dataset'] == dataset]

            x = np.arange(len(subset)) 
            width = 0.35

            fig, ax = plt.subplots(figsize=(3.2, 2.0))
            ax.bar(x - width/2, subset[f'Classical {metric_choice_bar}'], width, label='Classical', color='steelblue')
            ax.bar(x + width/2, subset[f'Quantum {metric_choice_bar}'], width, label='Quantum', color='orange')

            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(d)}%" for d in subset['Data Used']], rotation=45, fontsize=7)
            ax.set_ylabel(metric_choice_bar, fontsize=7)
            ax.set_xlabel("Data Used (%)", fontsize=7)
            ax.set_title(f"{dataset}", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(pad=0.3)

            with cols[col_index]:
                st.pyplot(fig, use_container_width=False)

            col_index += 1
            if col_index == 2:
                cols = st.columns(2)
                col_index = 0
            plt.close(fig)

        if num_datasets % 2 != 0:
            last_dataset = datasets[-1]
            subset = summary_df[summary_df['Dataset'] == last_dataset]

            x = np.arange(len(subset))
            width = 0.35

            cols = st.columns([1, 2, 1])  
            fig, ax = plt.subplots(figsize=(3.2, 2.0))
            ax.bar(x - width/2, subset[f'Classical {metric_choice_bar}'], width, label='Classical', color='steelblue')
            ax.bar(x + width/2, subset[f'Quantum {metric_choice_bar}'], width, label='Quantum', color='orange')

            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(d)}%" for d in subset['Data Used']], rotation=45, fontsize=7)
            ax.set_ylabel(metric_choice_bar, fontsize=7)
            ax.set_xlabel("Data Used (%)", fontsize=7)
            ax.set_title(f"{last_dataset}", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(pad=0.3)

            with cols[1]: 
                st.pyplot(fig, use_container_width=False)

            plt.close(fig)

    except Exception as e:
        st.error(f"⚠️ Could not load the results file: {e}")
