import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import faiss
import os
import time
from datetime import datetime
from sklearn.ensemble import IsolationForest

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

# --- Dark Mode & Theme Customization ---
st.set_page_config(page_title="AI Banking Assistant", layout="wide")

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if dark_mode:
    st.session_state.dark_mode = True
else:
    st.session_state.dark_mode = False

theme_css = """
    <style>
        body {
            background-color: #111827 !important;
            color: #F9FAFB !important;
        }
        .stButton>button {
            background-color: #10B981 !important;
            color: white !important;
        }
    </style>
""" if st.session_state.dark_mode else ""
st.markdown(theme_css, unsafe_allow_html=True)

# --- UI Sidebar Navigation ---
with st.sidebar:
    st.title("🔹 Navigation")
    selected_section = st.radio(
        "Go to:", 
        ["📂 Upload Data", "🚨 Customer Service & Fraud Detection", "🤖 AI Chat Assistant"],
        index=0
    )

    # Move File Upload to Sidebar (Global Scope)
    uploaded_file = st.file_uploader("Upload an Excel file", type=["csv"])

    # Store the uploaded file in session state so it is accessible across all sections
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

# --- Load Data Function ---
def load_data():
    """Loads the uploaded file and stores it globally in session state."""
    if "uploaded_file" in st.session_state:
        try:
            df = pd.read_csv(st.session_state["uploaded_file"])
            st.session_state["df"] = df
            return df
        except Exception as e:
            st.error(f"❌ Failed to load CSV file. Error: {str(e)}")
            return None
    return None

# --- SECTION 1: File Upload ---
if selected_section == "📂 Upload Data":
    st.title("📂 Upload Banking Data")

    df = load_data()  # Ensure data is loaded

    if df is not None:
        st.success("✅ File uploaded successfully!")
        st.write(df.head())

# --- SECTION 2: Customer Service Issues & Fraud Detection ---
if selected_section == "🚨 Customer Service & Fraud Detection":
    st.title("🚨 AI-Powered Fraud Detection & Customer Issue Resolution")

    df = load_data()  # Ensure data is loaded

    if df is not None and 'issue_type' in df.columns:
        # Display Summary of Customer Service Issues
        st.subheader("📊 Customer Complaints Summary")
        issue_counts = df["issue_type"].value_counts()
        st.bar_chart(issue_counts)

        # AI Insights on Customer Complaints
        st.subheader("🤖 AI Analysis of Customer Complaints")
        complaints_summary_prompt = f"""
        Analyze the following customer service issues and provide key trends:
        {df[['issue_type', 'description']].head(10).to_string(index=False)}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": complaints_summary_prompt}]
        )

        st.write(response.choices[0].message.content)

        # Fraud Detection Using Isolation Forest
        if 'account_balance' in df.columns and 'num_transactions_monthly' in df.columns:
            st.subheader("🚨 Fraud Detection Model")

            fraud_model = IsolationForest(contamination=0.05, random_state=42)
            df['fraud_risk'] = fraud_model.fit_predict(df[['account_balance', 'num_transactions_monthly']])

            fraud_cases = df[df['fraud_risk'] == -1]
            st.warning(f"🚨 {len(fraud_cases)} Potential Fraud Cases Detected!")

            if not fraud_cases.empty:
                st.write(fraud_cases[['customer_id', 'full_name', 'account_balance', 'num_transactions_monthly']])

        else:
            st.error("❌ Fraud detection requires 'account_balance' and 'num_transactions_monthly' columns.")

    else:
        st.warning("📂 Please upload a valid dataset with customer service issues.")

# --- SECTION 3: AI Chat Assistant (Enhanced UI) ---
if selected_section == "🤖 AI Chat Assistant":
    st.title("🤖 AI Chat Assistant (RAG)")

    df = load_data()  # Ensure data is loaded

    # Store Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User Input Box for Chat
    user_question = st.text_input("💬 Ask a banking-related question:")

    if st.button("🚀 Ask AI"):
        if user_question and df is not None:
            # Show Loading Indicator
            with st.spinner("🤖 Thinking..."):
                time.sleep(1)  # Simulate loading time
                query_vector = embed_text(user_question)
                retrieved_data = df.head(5).to_string(index=False)  # Send first 5 rows as reference
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Use the retrieved banking data below to answer questions."},
                        {"role": "user", "content": f"{retrieved_data}\n\n{user_question}"}
                    ]
                )
                
                # Store in chat history with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append(("user", "🧑‍💼 You", user_question, timestamp))
                st.session_state.chat_history.append(("ai", "🤖 AI", response.choices[0].message.content, timestamp))

    # Display Chat History with Correct Icons
    st.subheader("📜 Chat History")
    for role, sender, message, timestamp in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(f"**{sender}** ({timestamp})")
            st.write(message)
