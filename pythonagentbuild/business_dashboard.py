import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import faiss
import os
import time
from datetime import datetime

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
st.set_page_config(page_title="CEO Business Dashboard", layout="wide")

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
        .stChatMessage {
            padding: 10px;
            margin-bottom: 5px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #6B7280 !important;
            color: white !important;
        }
        .ai-message {
            background-color: #2563EB !important;
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
        ["📂 Upload Data", "📊 Revenue Forecast & KPIs", "🤖 AI Chat Assistant"],
        index=0
    )

    # Move File Upload to Sidebar (Global Scope)
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# --- SECTION 3: AI Chat Assistant (Enhanced UI) ---
if selected_section == "🤖 AI Chat Assistant":
    st.title("🤖 AI Chat Assistant (RAG)")

    # Store Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User Input Box for Chat
    user_question = st.text_input("💬 Ask a business-related question:")

    if st.button("🚀 Ask AI"):
        if user_question:
            # Show Loading Indicator
            with st.spinner("🤖 Thinking..."):
                time.sleep(1)  # Simulate loading time
                query_vector = embed_text(user_question)
                retrieved_data = "No relevant data found in uploaded files." if index.ntotal == 0 else list(knowledge_base.keys())[0]
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Use the retrieved data below to answer questions."},
                        {"role": "user", "content": f"{retrieved_data}\n\n{user_question}"}
                    ]
                )
                
                # Store in chat history with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append(("🧑‍💼 You", user_question, timestamp))
                st.session_state.chat_history.append(("🤖 AI", response.choices[0].message.content, timestamp))

    # Display Chat History with Timestamps
    st.subheader("📜 Chat History")
    for sender, message, timestamp in st.session_state.chat_history:
        with st.chat_message(sender):
            st.write(f"**{sender}** ({timestamp})")
            st.write(message)
