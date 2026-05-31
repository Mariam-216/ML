import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

def model_selection_page():
    
    st.header(" Model Selection & Training ")

    # VALIDATION CHECK.

    if "data_processed" not in st.session_state or st.session_state["data_processed"] is None:
        st.error("Didn't find the processed DATA ! Please make sure activating preprocessing page FIRST.")
        return
    else:
        df = st.session_state["data_processed"]
    
    # RETRIVE PREPROCESSED DATAFRAME FOR MODELING.

    df = st.session_state ["data_processed"]
    st.write(f"Dataset Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # SELECT MACHINE LEARNING TYPE.

    ml_type = st.radio("Select Task Type",["Classification" , "Regression" , "Clustering"])
    model = None
    X = None
    Y = None
    model_name = ""

    # CLASSIFICATION (1).

    if ml_type == "Classification":

        model_name = st.selectbox("Choose Algorithm",
                                  ["Logistic Regression","Random Forest","SVM","k-Nearest Neighbors","Naive Bayes","Neural Networks", "Decision Tree"])
        target = st.selectbox("Select Target Column" , df.columns)
        X = df.drop(columns=[target])
        Y = df[target]

        if model_name == "Logistic Regression": model = LogisticRegression()
        elif model_name == "SVM": model = SVC()
        elif model_name == "Random Forest": model = RandomForestClassifier()
        elif model_name == "k-Nearest Neighbors": model = KNeighborsClassifier()
        elif model_name == "Naive Bayes": model = GaussianNB()
        elif model_name == "Neural Networks": model = MLPClassifier(max_iter = 500)
        elif model_name == "Decision Tree": 
            max_depth = st.slider("Select Max Depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth)

    # REGRESSION (2).

    elif ml_type == "Regression":

        target = st.selectbox("Select Target Column" , df.columns)
        X = df.drop(columns=[target])
        Y = df[target]
        model = LinearRegression()
        model_name = "LinearRegression"
        st.write("Selected: Linear Regression")

    # CLUSTERING (3).

    elif ml_type == "Clustering":
        
        k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)    # SLISER FOR CLUSTERS NO SELECTION.
        model = KMeans(n_clusters = k)
        X = df
        model_name = "K-Means"
        st.write(f"Selected: K-Means with {k} clusters (Unsupervised)")

    if st.button("Train Model"):
        try:
            X = X.select_dtypes(include=[np.number])   # ONLY NUMERIC FEATURES.
            
            if model is None:
                st.error(" Please select a model to train ! ")
                return
            
            with st.spinner('Training the model... please wait.'):
                if ml_type != "Clustering":
                    
                    X_train , X_test ,Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)    # SPLITTING DATA INTO TRAINING & TESTING (80/20 %).   
                    
                    model.fit(X_train , Y_train)   # FITTING MODEL.

                    # SAVING TRAINED & TEST IN SESSION *FOR U MEMBER 7 ;)* .
                    st.session_state["trained_model"] = model
                    st.session_state["X_test"] = X_test
                    st.session_state["Y_test"] = Y_test
                    st.session_state["Y_pred"] = model.predict(X_test)

                else:
                    model.fit(X)
                    st.session_state["trained_model"] = model

            st.success(f"{model_name} trained successfully !!")

        except Exception as e:
            st.error(f" Training Failed: {e} ")

if __name__ == "__main__":
    model_selection_page()
    
    # DONE , BESTLUCK MEMB.(7). 
