"""
Example: load the saved model and predict churn probability for new customers.
Run: python3 predict_new_customer.py
"""
import joblib
import pandas as pd

model = joblib.load("models/churn_model.joblib")

# Example new customers (must have the same columns as training data, minus Churn/customerID)
new_customers = pd.DataFrame([
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 191.0,
        "SupportTickets": 4,
    },
    {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
        "SupportTickets": 0,
    },
])

probabilities = model.predict_proba(new_customers)[:, 1]
predictions = model.predict(new_customers)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "WILL CHURN" if pred == 1 else "will stay"
    print(f"Customer {i+1}: {label}  (churn probability: {prob:.1%})")
