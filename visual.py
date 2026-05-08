import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualization_page():
    st.header(" Data Visualization")

    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning(" Please upload a dataset first on the Upload page.")
        return
    else:
       df = st.session_state["data"]
       filename = st.session_state.get("filename", "dataset")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(num_cols) == 0:
        st.error("No numeric columns found in the dataset.")
        return

    st.caption(
        f" **{filename}** — "
        f"{df.shape[0]} rows × {df.shape[1]} cols | "
        f" {len(num_cols)} numeric |  {len(cat_cols)} categorical"
    )
    with st.expander(" Preview data"):
        st.dataframe(df.head(30), use_container_width=True)

    st.markdown("---")
    sns.set_theme(style="whitegrid", palette="muted")

    plot_type = st.selectbox("Plot Type", ["Line Plot", "Scatter Plot", "Box Plot"])

    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        if plot_type == "Line Plot":
            st.markdown("*Best for: trends over an index or ordered column*")

            col1, col2, col3 = st.columns(3)
            with col1:
                x = st.selectbox("X axis", all_cols)
            with col2:
                y = st.selectbox("Y axis (numeric)", num_cols)
            with col3:
                hue = st.selectbox("Hue (optional)", ["None"] + cat_cols)

            hue_val = None if hue == "None" else hue

            if x in num_cols:
                plot_df = (df[[x, y] + ([hue_val] if hue_val else [])]
                           .dropna().sort_values(x))
                sns.lineplot(data=plot_df, x=x, y=y,
                             hue=hue_val, linewidth=2, ax=ax)
            else:
                plot_df = df[[x, y] + ([hue_val] if hue_val else [])].dropna()
                sns.pointplot(data=plot_df, x=x, y=y,
                              hue=hue_val, capsize=0.1, ax=ax)
                plt.xticks(rotation=30)
                st.caption("Categorical X — showing mean ± CI per category")

        elif plot_type == "Scatter Plot":
            st.markdown("*Best for: relationship between two numeric variables*")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x = st.selectbox("X axis (numeric)", num_cols)
            with col2:
                y = st.selectbox("Y axis (numeric)", num_cols,
                                 index=min(1, len(num_cols) - 1))
            with col3:
                hue = st.selectbox("Hue (optional)", ["None"] + cat_cols + num_cols)
            with col4:
                show_reg = st.checkbox("Regression line", value=False)

            hue_val = None if hue == "None" else hue
            plot_df = df[[x, y] + ([hue_val] if hue_val else [])].dropna()

            if show_reg and hue_val is None:
                sns.regplot(data=plot_df, x=x, y=y,
                            scatter_kws={"alpha": 0.5}, ax=ax)
            else:
                if show_reg:
                    st.caption("Regression line is disabled when hue is selected")
                sns.scatterplot(data=plot_df, x=x, y=y,
                                hue=hue_val, alpha=0.7, ax=ax)

            if x != y:
                corr = df[[x, y]].dropna().corr().iloc[0, 1]
                strength = ("Strong" if abs(corr) > 0.7
                            else "Moderate" if abs(corr) > 0.4
                            else "Weak")
                st.metric("Pearson Correlation", f"{corr:.3f}", delta=strength)

        elif plot_type == "Box Plot":
            st.markdown("*Best for: distribution and outliers of numeric data*")

            col1, col2, col3 = st.columns(3)
            with col1:
                y_cols = st.multiselect(
                    "Numeric column(s)", num_cols, default=num_cols[:3]
                )
            with col2:
                x = st.selectbox("Group by category (optional)", ["None"] + cat_cols)
            with col3:
                show_outliers = st.checkbox("Show outliers", value=True)

            if not y_cols:
                st.warning("Select at least one numeric column.")
                plt.close(fig)
                return

            x_val = None if x == "None" else x
            flierprops = dict(marker="o", markerfacecolor="tomato",
                              markersize=4, alpha=0.6)
            plt.close(fig)

            if x_val is None:
                df_melt = df[y_cols].dropna().melt(
                    var_name="Variable", value_name="Value"
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(data=df_melt, x="Variable", y="Value",
                            showfliers=show_outliers,
                            flierprops=flierprops if show_outliers else {},
                            ax=ax)
                plt.xticks(rotation=20)
            else:
                n = len(y_cols)
                fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
                axes = [axes] if n == 1 else list(axes)
                for ax_i, col in zip(axes, y_cols):
                    plot_df = df[[x_val, col]].dropna()
                    sns.boxplot(data=plot_df, x=x_val, y=col,
                                showfliers=show_outliers,
                                flierprops=flierprops if show_outliers else {},
                                ax=ax_i)
                    ax_i.set_title(col, fontsize=12)
                    if df[x_val].nunique() > 6:
                        ax_i.tick_params(axis="x", rotation=45)
                fig.suptitle(f"Grouped by: {x_val}", fontsize=13, y=1.02)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("####  Descriptive Statistics")
            st.dataframe(df[y_cols].describe().round(3), use_container_width=True)
            return

        ax.set_title(plot_type, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Plotting error: {e}")
    finally:
        plt.close(fig)
if __name__ == "__main__":
    visualization_page()