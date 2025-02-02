import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('env_variables.env')

# Load the Excel File
file_path = "ceo_assistant_data (1).xlsx"
df = pd.read_excel(file_path)

# Get API key from environment variables
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Ensure 'Revenue' column exists
if 'Revenue' not in df.columns:
    raise ValueError("The dataset must contain a 'Revenue' column.")

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

def predict_future_revenue(months=6):
    """Predicts future revenue for the next N months."""
    future_months = np.arange(len(df), len(df) + months).reshape(-1, 1)
    predicted_revenue = model.predict(future_months)
    forecast_df = pd.DataFrame({"Month": future_months.flatten(), "Predicted Revenue": predicted_revenue})

    # Plot the predictions
    plt.figure(figsize=(10, 5))
    plt.scatter(df["Month"], df["Revenue"], color="blue", label="Actual Revenue")
    plt.plot(future_months, predicted_revenue, color="red", linestyle="dashed", label="Predicted Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.title("Revenue Prediction for Next 6 Months")
    plt.legend()
    plt.grid()
    plt.show()

    return forecast_df.to_string(index=False)

def ask_ai_about_data(question):
    """Handles AI-generated responses, including revenue forecasting."""
    if "projected revenue" in question.lower() or "future revenue" in question.lower():
        return predict_future_revenue(6)  # Predict next 6 months

    # Convert DataFrame to a string summary
    data_summary = df.describe().to_string()

    # AI Analysis
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a business data expert. Use the data to answer questions."},
            {"role": "user", "content": f"Here is the business data:\n{data_summary}\n\nNow, answer this question: {question}"}
        ]
    )

    return response.choices[0].message.content

# Streamlit UI (Optional for a Web Interface)
import streamlit as st

st.title("💼 CEO AI Assistant")
st.write("Ask business-related questions based on company data.")

# Input Box
user_input = st.text_input("Enter your question:")

if st.button("Ask AI"):
    if user_input:
        answer = ask_ai_about_data(user_input)
        st.subheader("AI Response:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")

# Command-Line Chat Loop
if __name__ == "__main__":
    print("🔹 CEO AI Assistant Ready! Type 'exit' to quit.")
    while True:
        user_input = input("CEO: ")
        if user_input.lower() == "exit":
            print("AI: Goodbye!")
            break
        answer = ask_ai_about_data(user_input)
        print("AI:", answer)
