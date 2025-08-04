import pandas as pd
import os
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Import the predict_cluster function
from modules.predictor import predict_cluster

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'modules', 'model.pkl')
if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure 'model.pkl' exists in the 'modules' folder.")
    st.stop()
model = joblib.load(model_path)

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'Data', 'mall_customers (1).csv')
if not os.path.exists(data_path):
    st.error("Dataset file not found. Please ensure 'mall_customers (1).csv' exists in the 'Data' folder.")
    st.stop()
data = pd.read_csv(data_path)

# Streamlit App
st.title("Mall Customers Clustering")
st.write("Dataset Preview:")
st.dataframe(data)

# Visualization: Pairplot
st.write("### Data Distribution")
try:
    fig = sns.pairplot(data[['Age', 'Annual_Income', 'Spending_Score']])
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error generating pairplot: {e}")

# Visualization: Cluster Scatter Plot
st.write("### Clustering Visualization")
try:
    data['Cluster'] = model.predict(data[['Age', 'Annual_Income', 'Spending_Score']])
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        data['Annual_Income'], 
        data['Spending_Score'], 
        c=data['Cluster'], 
        cmap='viridis', 
        alpha=0.7
    )
    ax.set_title("Clusters based on Annual Income and Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error generating scatter plot: {e}")

# Input form for cluster prediction
st.write("### Predict Cluster")
age = st.number_input("Age", min_value=0, max_value=100, value=25)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict"):
    try:
        # Call the predict_cluster function
        prediction = predict_cluster(model, age, income, spending_score)
        st.write(f"Predicted Cluster: {prediction}")
    except Exception as e:
        st.error(f"Error predicting cluster: {e}")
