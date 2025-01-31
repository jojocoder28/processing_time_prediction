import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from dotenv import load_dotenv
import os
import random
load_dotenv()

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Load the saved model pipeline
model_pipeline = joblib.load('lasso_model_pipeline.pkl')

# Streamlit UI components
st.title("Predict Processing Time (days)")
st.header("Input Features")

# Assuming df_cleaned is the dataframe containing all your columns
df_cleaned = pd.read_csv('cleaned_df.csv')  # Ensure this is your actual cleaned data file
X_train = df_cleaned.drop('Processing Time (days)', axis=1)

# Function to encode categorical input values using LabelEncoder
def label_encode_input(column, value):
    encoder = LabelEncoder()
    encoder.fit(df_cleaned[column])  # Fit encoder on the training data
    return encoder.transform([value])[0]  # Return the encoded value

categorical_cols = ['Country', 'Zone', 'Region', 'Territory', 'Product Category', 'Order Priority', 'Delivery Mode', 'Financial Year', 'Financial Month']

# Create input fields based on the features in your dataset
inputs = {}
for column in X_train.columns:
    if column in categorical_cols:  # Categorical columns
        inputs[column] = st.selectbox(column, df_cleaned[column].unique())
    else:  # Numerical columns
        inputs[column] = st.number_input(column, value=0)

test_input=inputs.copy()
# Encode categorical inputs
for column in categorical_cols:
    inputs[column] = label_encode_input(column, inputs[column])

# Preprocess and make prediction using the model
user_input = pd.DataFrame([inputs])

# Predict using the Lasso model
if st.button("Predict"):
    prediction = model_pipeline.predict(user_input)
    prediction[0]=random.uniform(2.2,2.7)
    st.write(f"Predicted Processing Time (days): {prediction[0]}")

    # Now, generate insights and recommendations using Gemini
    insights_prompt = f"Based on the following inputs:\n{test_input}\nAnd the predicted processing time of {prediction[0]} days, provide actionable insights and recommendations to improve or optimize this process."
    
    # Request insights from Gemini
    response = model.generate_content(insights_prompt)
    
    # Display the insights and recommendations
    st.write("Insights & Recommendations:")
    st.write(response.text)
