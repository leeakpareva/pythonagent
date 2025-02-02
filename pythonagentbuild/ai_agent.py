import openai
import pandas as pd

# Set your OpenAI API Key
OPENAI_API_KEY = "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"

class AIAgent:
    def __init__(self, dataframe=None):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.df = dataframe
    
    def get_response(self, user_input):
        """Get basic response from OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
            
    def get_response_with_data(self, question):
        """Get response using the Excel data"""
        try:
            # Convert DataFrame to a string summary
            data_summary = self.df.describe().to_string()
            # Add column names and first few rows for context
            data_context = f"\nColumns: {', '.join(self.df.columns)}"
            data_context += f"\nFirst few rows:\n{self.df.head().to_string()}"

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business data expert. Use the provided data to answer questions accurately."},
                    {"role": "user", "content": f"Here is the business data summary:{data_summary}\n{data_context}\n\nQuestion: {question}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

def simple_ai_agent():
    openai.api_key = OPENAI_API_KEY
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("AI: Goodbye!")
            break

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Change to "gpt-4" if available
            messages=[{"role": "user", "content": user_input}]
        )

        print("AI:", response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    simple_ai_agent()
