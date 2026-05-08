import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.cluster import KMeans

def evaluation_page():
    st.header("Model Evaluation")

    #  Validation 
    if "trained_model" not in st.session_state or st.session_state["trained_model"] is None:
        st.error(" No trained model found! Please go to **Model Selection** and train a model first.")
        return

    model = st.session_state["trained_model"]
    model_name = type(model).__name__
    is_clustering = isinstance(model, KMeans)

    if not is_clustering:
        if "X_test" not in st.session_state or st.session_state["X_test"] is None:
            st.error("Test data not found. Please retrain the model from the **Model Selection** page.")
            return
        Y_test = st.session_state["Y_test"]
        Y_pred = st.session_state["Y_pred"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#808080,#BABBB9);
                border-radius:12px;padding:16px 24px;margin-bottom:20px;color:white;">
        <h4 style="margin:0;color:white;">Active Model: {model_name}</h4>
        <p style="margin:4px 0 0 0;opacity:0.85;font-size:0.9em;">
            Results update automatically whenever you train a new model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sns.set_theme(style="whitegrid", palette="muted")

    # Clustering Evaluation
    if is_clustering:
        st.subheader(" Clustering Evaluation (K-Means)")

        data = st.session_state.get("data_processed")
        if data is None:
            st.error("Processed data not found.")
            return

        X = data.select_dtypes(include=[np.number])
        labels = model.labels_
        k = model.n_clusters
        unique, counts = np.unique(labels, return_counts=True)

        # Silhouette Score
        try:
            sil_score = silhouette_score(X, labels)
        except Exception:
            sil_score = None

        #  Metrics Row 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Number of Clusters (k)", k)
        col2.metric("Largest Cluster Size", int(counts.max()))
        col3.metric("Smallest Cluster Size", int(counts.min()))
        if sil_score is not None:
            col4.metric(" Silhouette Score", f"{sil_score:.4f}",
                        help="Ranges from -1 to 1. Closer to 1 = better-defined clusters.")
        
        # Silhouette explanation
        if sil_score is not None:
            if sil_score >= 0.7:
                quality = (" Strong", "Excellant Clustering")
            elif sil_score >= 0.5:
                quality = (" Reasonable", "Good Clustering")
            elif sil_score >= 0.25:
                quality = (" Weak", " you may need to change the k")
            else:
                quality = (" Poor", "Try another numbers of clustering")
            st.caption(f"Silhouette Quality: **{quality[0]}** — {quality[1]}")

        # Cluster Size Distribution
        st.markdown("####  Cluster Size Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([f"Cluster {i}" for i in unique], counts,
               color=sns.color_palette("pastel", k))
        ax.set_ylabel("Number of Samples")
        ax.set_title("Number of Samples per Cluster")
        for i, count in enumerate(counts):
            ax.text(i, count + 0.5, str(count), ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)
        return

    # Regression Evaluation
    if "Regress" in model_name:
        st.subheader(" Regression Evaluation")

        mse  = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(Y_test, Y_pred)

        # Metrics Row 
        c1, c2, c3 = st.columns(3)
        c1.metric(" R² Score",  f"{r2:.4f}",
                  help="Closer to 1 = model explains the data well.")
        c2.metric(" MSE",  f"{mse:.4f}",
                  help="Mean Squared Error — lower is better.")
        c3.metric(" RMSE", f"{rmse:.4f}",
                  help="Root Mean Squared Error — same unit as target.")

        # R² quality hint
        if r2 >= 0.9:
            st.success(f"R² = {r2:.4f} — Excellent fit! The model explains the data very well.")
        elif r2 >= 0.7:
            st.info(f"R² = {r2:.4f} — Good fit.")
        elif r2 >= 0.5:
            st.warning(f"R² = {r2:.4f} — Moderate fit. Consider more feature engineering.")
        else:
            st.error(f"R² = {r2:.4f} — Poor fit. The model struggles to explain the variance.")

        #  Actual vs Predicted 
        st.markdown("####  Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(Y_test, Y_pred, alpha=0.6, color="#667eea", edgecolors="white", s=50)
        lims = [min(float(Y_test.min()), float(Y_pred.min())),
                max(float(Y_test.max()), float(Y_pred.max()))]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        #  Residuals Distribution 
        st.markdown("####  Residuals Distribution")
        residuals = np.array(Y_test) - np.array(Y_pred)
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(Y_pred, residuals, alpha=0.6, color="#764ba2", s=40)
        axes[0].axhline(0, color="tomato", linestyle="--")
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        sns.histplot(residuals, kde=True, ax=axes[1], color="#667eea")
        axes[1].set_title("Residuals Distribution")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        return

    # Classification Evaluation
    st.subheader(" Classification Evaluation")

    acc     = accuracy_score(Y_test, Y_pred)
    classes = np.unique(Y_test)
    report  = classification_report(Y_test, Y_pred, output_dict=True)
    avg     = report.get("weighted avg", {})

    # Key Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" Accuracy",  f"{acc:.4f}")
    c2.metric(" Precision", f"{avg.get('precision', 0):.4f}")
    c3.metric(" Recall",    f"{avg.get('recall', 0):.4f}")
    c4.metric(" F1-Score",  f"{avg.get('f1-score', 0):.4f}")

    # Classification Report
    st.markdown("####  Classification Report")
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(
        report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
        use_container_width=True
    )

    # Confusion Matrix
    st.markdown("####  Confusion Matrix")
    cm  = confusion_matrix(Y_test, Y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

evaluation_page()