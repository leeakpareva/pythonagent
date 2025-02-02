import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import faiss
import os

# Securely fetch the OpenAI API key from Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key is missing. Please set it in Streamlit Secrets.")
else:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

# FAISS Setup (Using OpenAI Embeddings)
vector_dim = 1536  # OpenAI's text-embedding-ada-002 model dimension
index = faiss.IndexFlatL2(vector_dim)
knowledge_base = {}

# Function to Get Embeddings from OpenAI
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# Title
st.title("📊 CEO Business Dashboard (RAG-Powered)")
st.write("Upload your business data, get revenue forecasts, and chat with your AI Assistant.")

# File Upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    # Try loading the Excel file with openpyxl
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ Failed to load Excel file. Error: {str(e)}")
        st.stop()

    # Display raw data
    st.subheader("📂 Uploaded Data Preview")
    st.write(df.head())

    # Convert Data to Text & Store in FAISS
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
        vector = embed_text(row_text)
        index.add(np.expand_dims(vector, axis=0))
        knowledge_base[str(row_text)] = vector

    # Ensure 'Revenue' column exists
    if 'Revenue' in df.columns:
        # Clean Data
        df = df.dropna(subset=['Revenue'])
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')

        # Add a time index
        df["Month"] = np.arange(len(df))

        # Train ML Model
        X = df[["Month"]]
        y = df["Revenue"]
        model = LinearRegression()
        model.fit(X, y)

        # Predict Future Revenue
        future_months = np.arange(len(df), len(df) + 6).reshape(-1, 1)
        predicted_revenue = model.predict(future_months)
        forecast_df = pd.DataFrame({"Month": future_months.flatten(), "Predicted Revenue": predicted_revenue})

        # Plot Actual vs Predicted Revenue
        st.subheader("📈 Revenue Forecast (Next 6 Months)")
        fig = px.line(
            x=list(df["Month"]) + list(future_months.flatten()), 
            y=list(df["Revenue"]) + list(predicted_revenue), 
            labels={"x": "Month", "y": "Revenue"},
            title="Revenue Trend & Forecast"
        )
        fig.add_scatter(x=df["Month"], y=df["Revenue"], mode="markers", name="Actual Revenue")
        st.plotly_chart(fig)

        # Display Forecast Data
        st.subheader("📊 Forecasted Revenue")
        st.write(forecast_df)

    else:
        st.warning("The uploaded file must contain a 'Revenue' column for forecasting.")

# --- RAG-Powered AI Chat ---
st.subheader("💬 Chat with Your AI Assistant (RAG)")

# Input Box for Chat
user_question = st.text_input("Ask a business-related question:")

if st.button("Ask AI"):
    if user_question:
        # Embed Question
        query_vector = embed_text(user_question)
        
        # Retrieve Closest Data (Fixed FAISS Index Error)
        if index.ntotal > 0:  # Check if there is data in FAISS
            D, I = index.search(np.expand_dims(query_vector, axis=0), 1)  # Find closest match
            if I[0][0] == -1 or I[0][0] >= len(knowledge_base):  # Ensure valid index
                retrieved_data = "No relevant data found in uploaded files."
            else:
                retrieved_data = list(knowledge_base.keys())[I[0][0]]
        else:
            retrieved_data = "No relevant data available yet. Please upload a dataset."

        # Generate AI Response with Business Data
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business data expert. Use the retrieved data below to answer questions."},
                {"role": "user", "content": f"Here is the relevant data:\n{retrieved_data}\n\nNow, answer this question: {user_question}"}
            ]
        )

        # Display AI's Response
        st.subheader("🤖 AI Response:")
        st.write(response.choices[0].message.content)
    else:
        st.warning("Please enter a question before clicking 'Ask AI'.")
