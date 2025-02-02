import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the Excel file
file_path = "ceo_assistant_data (1).xlsx"
df = pd.read_excel(file_path)

# Ensure the dataset has a 'Revenue' column
if 'Revenue' not in df.columns:
    raise ValueError("The dataset must contain a 'Revenue' column.")

# Generate time-based index (assuming monthly data)
df["Month"] = np.arange(len(df))

# Select features (X) and target (y)
X = df[["Month"]]  # Independent variable (Time)
y = df["Revenue"]  # Dependent variable (Revenue)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict revenue for the next 6 months
future_months = np.arange(len(df), len(df) + 6).reshape(-1, 1)
predicted_revenue = model.predict(future_months)

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({"Month": future_months.flatten(), "Predicted Revenue": predicted_revenue})

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(df["Month"], df["Revenue"], color="blue", label="Actual Revenue")
plt.plot(future_months, predicted_revenue, color="red", linestyle="dashed", label="Predicted Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.title("Revenue Prediction for Next 6 Months")
plt.legend()
plt.grid()
plt.show()

# Print forecasted revenue
print("📊 Predicted Revenue for Next 6 Months:")
print(forecast_df)
