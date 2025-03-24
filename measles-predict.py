import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
vax_data = pd.read_csv('data/mmr-vax.csv')


# Sample dataset with multiple features


# Define X (independent variables) and y (dependent variable)
# X is often called the "features" and y is often called the "target."
X = vax_data[["Perc", "Population", "Area"]]  # Features
y = vax_data["Cases"]  # Target variable

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
new_year = [[94.3, 31290831, 268596.46]]
predicted_cases = model.predict(new_year)
print(f"Predicted cases: {predicted_cases[0]}")


# STUDENTS: you will peruse the files named lin.py and
# multi-lin.py to see examples of how to construct a linear regression
# model.  Then, you will try to create your own linear regression in this file
# that uses the data we have merged together (data/mmr-vax.csv)
# to predict the likelihood of a measles outbreak in U.S. states.


