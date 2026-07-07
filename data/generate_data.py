"""
Generates a realistic synthetic telecom customer churn dataset.
Mimics the structure of the popular Telco Customer Churn dataset,
with churn probability driven by real underlying patterns (contract type,
tenure, monthly charges, support tickets, etc.) so the ML pipeline has
genuine signal to learn from.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 7000

customer_id = [f"CUST-{10000+i}" for i in range(N)]
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

tenure = np.random.gamma(shape=2.0, scale=15, size=N).astype(int)
tenure = np.clip(tenure, 0, 72)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20]
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20]
)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.45, 0.55])
)

def dep_service(base_p_yes):
    return np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], N, p=[base_p_yes, 1 - base_p_yes])
    )

online_security = dep_service(0.35)
online_backup = dep_service(0.4)
device_protection = dep_service(0.4)
tech_support = dep_service(0.35)
streaming_tv = dep_service(0.45)
streaming_movies = dep_service(0.45)

paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.35, 0.2, 0.22, 0.23]
)

base_charge = np.where(internet_service == "Fiber optic", 70,
              np.where(internet_service == "DSL", 45, 20))
addon_cost = (
    (online_security == "Yes").astype(int) * 5 +
    (online_backup == "Yes").astype(int) * 5 +
    (device_protection == "Yes").astype(int) * 4 +
    (tech_support == "Yes").astype(int) * 5 +
    (streaming_tv == "Yes").astype(int) * 8 +
    (streaming_movies == "Yes").astype(int) * 8 +
    (phone_service == "Yes").astype(int) * 10
)
monthly_charges = base_charge + addon_cost + np.random.normal(0, 5, N)
monthly_charges = np.clip(monthly_charges, 18, 120).round(2)

total_charges = (monthly_charges * tenure + np.random.normal(0, 20, N)).clip(min=0).round(2)

# Support tickets - more tickets = more frustration = more likely to churn
support_tickets = np.random.poisson(1.2, N)
support_tickets = np.clip(support_tickets, 0, 10)

# --- Build churn probability from real underlying logic ---
logit = -1.5
logit += np.where(contract == "Month-to-month", 1.4, np.where(contract == "One year", 0.1, -1.0))
logit += -0.04 * tenure
logit += np.where(internet_service == "Fiber optic", 0.5, 0.0)
logit += 0.02 * (monthly_charges - 60)
logit += 0.18 * support_tickets
logit += np.where(tech_support == "No", 0.3, 0.0)
logit += np.where(online_security == "No", 0.2, 0.0)
logit += np.where(paperless_billing == "Yes", 0.15, 0.0)
logit += np.where(senior_citizen == 1, 0.2, 0.0)
logit += np.where(partner == "No", 0.15, 0.0)
logit += np.where(payment_method == "Electronic check", 0.25, 0.0)
logit += np.random.normal(0, 0.6, N)  # noise

prob_churn = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(N) < prob_churn).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SupportTickets": support_tickets,
    "Churn": churn_label,
})

# Introduce a few realistic missing values (like the real Telco dataset does for TotalCharges)
missing_idx = df.sample(frac=0.005, random_state=1).index
df.loc[missing_idx, "TotalCharges"] = np.nan

out_path = "/home/claude/churn_project/data/customer_churn.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} rows -> {out_path}")
print(df["Churn"].value_counts(normalize=True))
