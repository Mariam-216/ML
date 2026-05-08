import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def preprocessing_page():
    if st.session_state["data"] is None:
        st.warning("Upload a file first.")
    else:
        df = st.session_state["data"].copy()
    st.title("Preprocessing Page")

    # =========================
    # Preprocessing
    # =========================
    if st.session_state["data"] is not None:
        st.divider()
        st.header("Data Preprocessing & Transformation")

        df = st.session_state["data"].copy()

        # =========================
        # 1. Missing Values
        # =========================
        st.subheader("1. Handling Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()

        if missing_cols:
            st.warning(f"Columns with missing values: {', '.join(missing_cols)}")

            impute_method = st.selectbox(
                "Select Imputation Method:",
                ["None", "Simple Imputer (Mean)", "Simple Imputer (Median)", 
                 "Simple Imputer (Most Frequent)", "KNN Imputer", "Iterative Imputer"]
            )

            if impute_method != "None":
                num_missing = df[missing_cols].select_dtypes(include=np.number).columns
                cat_missing = df[missing_cols].select_dtypes(exclude=np.number).columns

                if len(num_missing) > 0:
                    if impute_method == "Simple Imputer (Mean)":
                        imputer = SimpleImputer(strategy='mean')
                    elif impute_method == "Simple Imputer (Median)":
                        imputer = SimpleImputer(strategy='median')
                    elif impute_method == "KNN Imputer":
                        imputer = KNNImputer(n_neighbors=5)
                    elif impute_method == "Iterative Imputer":
                        imputer = IterativeImputer(max_iter=10, random_state=0)

                    if impute_method != "Simple Imputer (Most Frequent)":
                        df[num_missing] = imputer.fit_transform(df[num_missing])

                if impute_method == "Simple Imputer (Most Frequent)":
                    imputer_freq = SimpleImputer(strategy='most_frequent')
                    df[missing_cols] = imputer_freq.fit_transform(df[missing_cols])
                elif len(cat_missing) > 0:
                    imputer_freq = SimpleImputer(strategy='most_frequent')
                    df[cat_missing] = imputer_freq.fit_transform(df[cat_missing])

                st.success(f"Missing values handled using {impute_method}.")
        else:
            st.info("No missing values found.")

        # =========================
        # 2. Outliers
        # =========================
        st.subheader("2. Outlier Detection & Handling")
        num_cols_for_outliers = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols_for_outliers:
            outlier_method = st.selectbox(
                "Select Outlier Handling Method:",
                ["None", "IQR (Clipping)", "Z-Score (Clipping)", 
                 "Winsorization", "Drop Outliers (IQR)"]
            )

            if outlier_method != "None":
                if outlier_method == "IQR (Clipping)":
                    for col in num_cols_for_outliers:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

                elif outlier_method == "Drop Outliers (IQR)":
                    for col in num_cols_for_outliers:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

                elif outlier_method == "Z-Score (Clipping)":
                    for col in num_cols_for_outliers:
                        mean = df[col].mean()
                        std = df[col].std()
                        df[col] = np.clip(df[col], mean - 3*std, mean + 3*std)

                elif outlier_method == "Winsorization":
                    for col in num_cols_for_outliers:
                        df[col] = winsorize(df[col], limits=[0.05, 0.05])

                st.success(f"Outliers handled using {outlier_method}")
                st.write(f"Shape: {df.shape}")

        # =========================
        # Feature Engineering & Selection
        # =========================
        st.divider()
        st.header("3. Feature Engineering & Selection")

        # Balancing
        st.subheader("A. Handling Imbalanced Data")
        target_for_balance = st.selectbox("Select Target for Balancing (Classification only):", [None] + list(df.columns))

        if target_for_balance:
            balance_method = st.radio("Select Balancing Method:", ["None", "Oversampling (SMOTE)", "Undersampling"])
            if balance_method != "None":
                X_temp = df.drop(columns=[target_for_balance])
                y_temp = df[target_for_balance]
                X_temp_dummy = pd.get_dummies(X_temp, drop_first=True)

                if balance_method == "Oversampling (SMOTE)":
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X_temp_dummy, y_temp)
                else:
                    rus = RandomUnderSampler(random_state=42)
                    X_res, y_res = rus.fit_resample(X_temp_dummy, y_temp)

                df = pd.concat([X_res, y_res], axis=1)
                st.success(f"Data balanced using {balance_method}. New shape: {df.shape}")

        # PCA
        st.subheader("B. Dimensionality Reduction (PCA)")
        if st.checkbox("Apply PCA"):
            n_components = st.slider("Select number of components:", 1, min(df.shape[1], 10), 2)
            pca = PCA(n_components=n_components)
            num_cols_pca = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols_pca:
                pca_results = pca.fit_transform(df[num_cols_pca])
                pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)])
                df = pd.concat([pca_df, df.select_dtypes(exclude=np.number).reset_index(drop=True)], axis=1)
                st.write(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")
                st.success("PCA Applied successfully.")

                # =========================
        # 3. Feature Transformation
        # =========================
        st.subheader("3. Feature Transformation")
        
        transform_cols = st.multiselect(
            "Select Numeric Columns to Transform:", 
            num_cols_for_outliers,
            help="Best used for skewed data like 'Fare'."
        )
        
        if transform_cols:
            transform_method = st.selectbox(
                "Select Transformation Method:",
                ["None", "Log Transformation (log1p)", "Box-Cox", "Yeo-Johnson (Power)", "Polynomial Features"]
            )

            if transform_method != "None":
                try:
                    if transform_method == "Log Transformation (log1p)":
                        df[transform_cols] = np.log1p(df[transform_cols])
                        st.success(f" Log transformation applied to {transform_cols}")

                    elif transform_method == "Box-Cox":
                        pt = PowerTransformer(method='box-cox')
                        df[transform_cols] = pt.fit_transform(df[transform_cols])
                        st.success(f" Box-Cox transformation applied to {transform_cols}")

                    elif transform_method == "Yeo-Johnson (Power)":
                        pt = PowerTransformer(method='yeo-johnson')
                        df[transform_cols] = pt.fit_transform(df[transform_cols])
                        st.success(f" Yeo-Johnson transformation applied to {transform_cols}")

                    elif transform_method == "Polynomial Features":
                        degree = st.slider("Select Polynomial Degree:", 2, 3, 2)
                        poly = PolynomialFeatures(degree=degree, include_bias=False)
                        
                        poly_data = poly.fit_transform(df[transform_cols])
                        poly_cols = poly.get_feature_names_out(transform_cols)
                        df_poly = pd.DataFrame(poly_data, columns=poly_cols, index=df.index)
                        
                        df = df.drop(columns=transform_cols)
                        df = pd.concat([df, df_poly], axis=1)
                        st.success(f" Generated {len(poly_cols)} Polynomial features")

                    st.dataframe(df.head())
                
                except Exception as e:
                    st.error(f" Transformation Error: {e}")
                    st.info("Tip: If you have zeros or negative numbers, use 'Yeo-Johnson' or 'Log' instead of Box-Cox.")

        # =========================
        # 4. Handling Imbalanced Data
        # =========================
        st.subheader("4. Handling Imbalanced Data")

        target_for_resampling = st.selectbox(
            "Select Target Column for Resampling:", 
            options=[None] + list(df.columns),
            help="Resampling will balance the classes in this column."
        )

        if target_for_resampling:
            class_counts = df[target_for_resampling].value_counts()
            st.write("Current Class Distribution:")
            st.bar_chart(class_counts)

            resample_method = st.radio(
                "Select Resampling Technique:",
                ["None", "Oversampling (SMOTE)", "Undersampling"]
            )

            if resample_method != "None":
                try:
                    X = df.drop(columns=[target_for_resampling])
                    y = df[target_for_resampling]
                    X_numeric = X.select_dtypes(include=[np.number])
                    
                    if resample_method == "Oversampling (SMOTE)":
                        if X_numeric.isnull().any().any():
                            st.error("Error: SMOTE does not accept missing values.")
                        else:
                            smote = SMOTE(random_state=42)
                            X_res, y_res = smote.fit_resample(X_numeric, y)
                            st.success("Oversampling complete using SMOTE.")

                    elif resample_method == "Undersampling":
                        rus = RandomUnderSampler(random_state=42)
                        X_res, y_res = rus.fit_resample(X_numeric, y)
                        st.success("Undersampling complete.")

                    # Reconstruct DF
                    df = pd.concat([X_res, y_res], axis=1)
                    st.write("New Class Distribution:")
                    st.bar_chart(df[target_for_resampling].value_counts())

                except Exception as e:
                    st.error(f"Resampling Error: {e}")

        # =========================
        # 5. Encoding & Scaling
        # =========================
        st.subheader("5. Encoding & Normalization")

        target_col = st.selectbox(
            "Select Target Column",
            options=[None] + list(df.columns),
            key="final_target"
        )

        feature_cols = [col for col in df.columns if col != target_col]
        num_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        cat_cols = df[feature_cols].select_dtypes(exclude=np.number).columns.tolist()

        exclude_scale = st.multiselect("Columns to EXCLUDE from Scaling", options=num_cols)
        exclude_encode = st.multiselect("Columns to EXCLUDE from Encoding", options=cat_cols)

        if st.button("Apply Transformation"):
            cols_to_scale = [c for c in num_cols if c not in exclude_scale]
            cols_to_encode = [c for c in cat_cols if c not in exclude_encode]

            df_final = pd.DataFrame()

            # Scaling
            if cols_to_scale:
                scaler = StandardScaler()
                df_scaled = pd.DataFrame(
                    scaler.fit_transform(df[cols_to_scale]),
                    columns=cols_to_scale,
                    index=df.index
                )
                df_final = pd.concat([df_final, df_scaled], axis=1)

            # Encoding
            if cols_to_encode:
                df_encoded = pd.get_dummies(df[cols_to_encode], drop_first=True)
                df_encoded.index = df.index
                df_final = pd.concat([df_final, df_encoded], axis=1)

            # Remaining
            remaining = [c for c in feature_cols if c not in cols_to_scale + cols_to_encode]
            if remaining:
                df_final = pd.concat([df_final, df[remaining]], axis=1)

            if target_col:
                df_final[target_col] = df[target_col]

            st.session_state["data_processed"] = df_final
            st.success("Processing Complete")
            st.dataframe(df_final.head())

    else:
        st.warning("Upload a file first.")

preprocessing_page()
