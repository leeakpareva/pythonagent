import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import faiss
import os
import time
import yaml
import streamlit_authenticator as stauth
from datetime import datetime
from sklearn.ensemble import IsolationForest
from yaml.loader import SafeLoader

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

# --- Authentication Setup ---
with open('credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

name, authentication_status, username = authenticator.login("🔒 Login", "main")

if authentication_status:
    authenticator.logout("🔓 Logout", "sidebar")
    st.sidebar.write(f"👤 Logged in as: **{name}**")

    # --- Dark Mode Fix & Theme Customization ---
    st.set_page_config(page_title="AI Banking Assistant", layout="wide")

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    if st.sidebar.checkbox("🌙 Enable Dark Mode", value=st.session_state.dark_mode):
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
            ["📂 Upload Data", "🚨 Customer Service & Fraud Detection", "📊 Banking Insights", "🤖 AI Chat Assistant", "ℹ️ About"],
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

    # --- SECTION 3: Banking Insights ---
    if selected_section == "📊 Banking Insights":
        st.title("📊 AI-Powered Banking Analytics")

    # --- SECTION 4: AI Chat Assistant ---
    if selected_section == "🤖 AI Chat Assistant":
        st.title("🤖 AI Chat Assistant (RAG)")

    # --- SECTION 5: About Page ---
    if selected_section == "ℹ️ About":
        st.title("ℹ️ About the AI Banking Assistant")

        st.markdown("""
        ## **What is This AI?**
        This is an **AI-powered banking assistant** that helps financial institutions manage customer data, detect fraud, and analyze financial trends.

        ## **How Does It Work?**
        - 📂 **Uploads & Analyzes Banking Data**  
        - 🚨 **Detects Fraudulent Transactions**  
        - 🤖 **Uses AI to Answer Banking-Related Questions**  
        - 📊 **Provides Financial Insights & KPIs**  

        ## **Why Was This Built?**
        - **Automate Banking Operations & Fraud Detection**  
        - **Enhance Customer Service Response**  
        - **Improve Financial Insights for Customers & Institutions**  

        **Future Upgrades Coming Soon!** 🚀
        """)

elif authentication_status == False:
    st.error("❌ Incorrect username or password.")

elif authentication_status == None:
    st.warning("Please enter your credentials to log in.")
