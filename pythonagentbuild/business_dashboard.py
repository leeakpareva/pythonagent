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

# --- Dark Mode & Theme Customization ---
st.set_page_config(page_title="CEO Business Dashboard", layout="wide")

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
theme_css = """
    <style>
        :root {
            --background: #111827;
            --foreground: #F9FAFB;
            --primary: #10B981;
            --secondary: #6B7280;
            --accent: #6366F1;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: var(--background) !important;
            color: var(--foreground) !important;
        }
        .stButton>button {
            background-color: var(--primary) !important;
            color: white !important;
        }
    </style>
""" if dark_mode else ""
st.markdown(theme_css, unsafe_allow_html=True)

# --- UI Sidebar Navigation ---
with st.sidebar:
    st.title("🔹 Navigation")
    selected_section = st.radio(
        "Go to:", 
        ["📂 Upload Data", "📊 Revenue Forecast & KPIs", "💬 AI Chat Assistant"],
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
        except Exception as e:
            st.error(f"❌ Failed to load Excel file. Error: {str(e)}")

# --- SECTION 2: Revenue Forecast & KPIs ---
if selected_section == "📊 Revenue Forecast & KPIs":
    st.title("📊 Revenue Forecast & Key Metrics")

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

            # --- KPI Section ---
            st.subheader("📌 Key Performance Indicators (KPIs)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📈 Total Revenue", f"${df['Revenue'].sum():,.2f}")

            with col2:
                st.metric("📊 Highest Revenue Month", f"Month {df['Revenue'].idxmax()}")

            with col3:
                st.metric("📉 Average Monthly Revenue", f"${df['Revenue'].mean():,.2f}")

            # --- Plot Forecast ---
            st.subheader("📈 Revenue Forecast (Next 6 Months)")
            fig = px.line(
                x=list(df["Month"]) + list(future_months.flatten()), 
                y=list(df["Revenue"]) + list(predicted_revenue), 
                labels={"x": "Month", "y": "Revenue"},
                title="Revenue Trend & Forecast",
                markers=True
            )
            fig.add_scatter(x=df["Month"], y=df["Revenue"], mode="markers", name="Actual Revenue")
            st.plotly_chart(fig)

            st.write(forecast_df)

        else:
            st.warning("The uploaded file must contain a 'Revenue' column.")

    else:
        st.warning("📂 Please upload a file first in the 'Upload Data' section.")

# --- SECTION 3: AI Chat Assistant (Enhanced UI) ---
if selected_section == "💬 AI Chat Assistant":
    st.title("💬 AI Chat Assistant (RAG)")

    # Store Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User Input Box for Chat
    user_question = st.text_input("💬 Ask a business-related question:")

    if st.button("🚀 Ask AI"):
        if user_question:
            query_vector = embed_text(user_question)
            retrieved_data = "No relevant data found in uploaded files." if index.ntotal == 0 else list(knowledge_base.keys())[0]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Use the retrieved data below to answer questions."},
                    {"role": "user", "content": f"{retrieved_data}\n\n{user_question}"}
                ]
            )
            
            # Store in chat history
            st.session_state.chat_history.append(("🧑‍💼 You", user_question))
            st.session_state.chat_history.append(("🤖 AI", response.choices[0].message.content))

    # Display Chat History
    st.subheader("📜 Chat History")
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.write(message)
