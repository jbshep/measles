import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset with multiple features
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Years of experience
    "Education_Level": [2, 3, 3, 4, 4, 5, 5, 5, 6, 6],  # Education level (e.g., 2 = high school, 6 = PhD)
    "Certifications": [0, 1, 2, 2, 3, 3, 4, 5, 6, 7],  # Number of certifications
    "Salary": [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]  # Salary (target variable)
}

df = pd.DataFrame(data)

# Define X (independent variables) and y (dependent variable)
# X is often called the "features" and y is often called the "target."
X = df[["Experience", "Education_Level", "Certifications"]]  # Features
y = df["Salary"]  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Model Coefficients: {model.coef_}")  # Coefficients for each feature
print(f"Model Intercept: {model.intercept_}")  # Intercept value

# Predict salary for a candidate with:
# 12 years of experience, education level 5, and 3 certifications
new_candidate = [[12, 5, 3]]
predicted_salary = model.predict(new_candidate)
print(f"Predicted Salary: {predicted_salary[0]}")

