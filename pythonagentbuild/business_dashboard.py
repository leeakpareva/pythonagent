import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import faiss
import os

# Securely fetch OpenAI API key from Streamlit Secrets
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

# Function to Rank & Retrieve Top Business Data
from openai.embeddings_utils import cosine_similarity

def get_most_relevant_data(query_vector, knowledge_base, top_n=3):
    """Retrieve the most relevant business data based on similarity ranking."""
    if len(knowledge_base) == 0:
        return ["No relevant data available. Please upload a dataset."]
    
    similarities = [(cosine_similarity(query_vector, vec), text) for text, vec in knowledge_base.items()]
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)  # Sort by highest similarity
    return [text for _, text in similarities[:top_n]]  # Return top matches

# Function to Generate AI Response
def generate_ai_response(question, retrieved_data):
    """Generates a structured AI response using retrieved business data."""
    system_prompt = """
    You are a business analyst. Use the retrieved data to provide structured insights:
    
    1️⃣ **Key Business Insights**
    2️⃣ **Trend Analysis & Predictions**
    3️⃣ **Actionable Strategies & Recommendations**
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Data: {retrieved_data}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Function to Summarize Uploaded Data
def summarize_uploaded_data(df):
    """Summarizes uploaded business data and injects it into AI responses."""
    summary_prompt = f"""
    Summarize this business data with key trends, anomalies, and strategic insights:
    {df.head(10).to_string(index=False)}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    return response.choices[0].message.content

# --- UI Sidebar Navigation ---
st.set_page_config(page_title="CEO Business Dashboard", layout="wide")

with st.sidebar:
    st.title("🔹 Navigation")
    selected_section = st.radio(
        "Go to:", 
        ["📂 Upload Data", "📊 Revenue Forecast", "💬 AI Chat Assistant"],
        index=0
    )

    # Move File Upload to Sidebar (Global Scope)
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# --- SECTION 1: File Upload ---
if selected_section == "📂 Upload Data":
    st.title("📂 Upload Business Data")

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success("✅ File uploaded successfully!")
            st.write(df.head())

            # Summarize and Display Data Insights
            summary = summarize_uploaded_data(df)
            st.subheader("📊 Data Insights Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"❌ Failed to load Excel file. Error: {str(e)}")

# --- SECTION 2: Revenue Forecast ---
if selected_section == "📊 Revenue Forecast":
    st.title("📊 Revenue Forecast")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        
        if 'Revenue' in df.columns:
            df = df.dropna(subset=['Revenue'])
            df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
            df["Month"] = np.arange(len(df))

            from sklearn.linear_model import LinearRegression
            X = df[["Month"]]
            y = df["Revenue"]
            model = LinearRegression()
            model.fit(X, y)

            future_months = np.arange(len(df), len(df) + 6).reshape(-1, 1)
            predicted_revenue = model.predict(future_months)
            forecast_df = pd.DataFrame({"Month": future_months.flatten(), "Predicted Revenue": predicted_revenue})

            # Plot Forecast
            fig = px.line(x=list(df["Month"]) + list(future_months.flatten()), 
                          y=list(df["Revenue"]) + list(predicted_revenue), 
                          labels={"x": "Month", "y": "Revenue"}, title="Revenue Trend & Forecast")
            fig.add_scatter(x=df["Month"], y=df["Revenue"], mode="markers", name="Actual Revenue")
            st.plotly_chart(fig)

            st.write(forecast_df)

        else:
            st.warning("The uploaded file must contain a 'Revenue' column.")

    else:
        st.warning("📂 Please upload a file first in the 'Upload Data' section.")

# --- SECTION 3: AI Chat Assistant ---
if selected_section == "💬 AI Chat Assistant":
    st.title("💬 AI Chat Assistant (RAG)")

    # Input Box for Chat
    user_question = st.text_input("Ask a business-related question:")
    
    if st.button("Ask AI"):
        if user_question:
            query_vector = embed_text(user_question)
            retrieved_data = get_most_relevant_data(query_vector, knowledge_base, top_n=3)
            response = generate_ai_response(user_question, retrieved_data)
            st.subheader("🤖 AI Response:")
            st.write(response)
        else:
            st.warning("Please enter a question before clicking 'Ask AI'.")
