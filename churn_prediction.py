"""
Customer Churn Prediction
=========================
End-to-end ML pipeline using Pandas + Scikit-learn:
  1. Load & clean data
  2. Exploratory Data Analysis (saved as PNG charts)
  3. Preprocessing (encoding, scaling) via sklearn Pipeline + ColumnTransformer
  4. Train & compare multiple models (Logistic Regression, Random Forest, Gradient Boosting)
  5. Evaluate with accuracy, precision, recall, F1, ROC-AUC + confusion matrix
  6. Feature importance
  7. Save the best model to disk (joblib) for reuse

Run: python3 churn_prediction.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

DATA_PATH = "data/customer_churn.csv"
OUT_DIR = "outputs"
MODEL_DIR = "models"
RANDOM_STATE = 42

sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ----------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading and cleaning data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Drop ID column (not predictive)
df = df.drop(columns=["customerID"])

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# TotalCharges has some missing values -> impute with median later in pipeline
# (kept as numeric; ensure dtype is correct)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(f"\nChurn rate: {df['Churn'].mean():.2%}")

# ----------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Exploratory Data Analysis")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Churn distribution
df["Churn"].map({1: "Yes", 0: "No"}).value_counts().plot(
    kind="bar", ax=axes[0, 0], color=["#4C72B0", "#DD8452"]
)
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].set_xlabel("Churn")
axes[0, 0].set_ylabel("Count")

# Churn by contract type
ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
ct.columns = ["No", "Yes"]
ct.plot(kind="bar", ax=axes[0, 1], color=["#4C72B0", "#DD8452"])
axes[0, 1].set_title("Churn Rate (%) by Contract Type")
axes[0, 1].set_ylabel("Churn Rate (%)")
axes[0, 1].legend(title="Churn")

# Tenure distribution by churn
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             palette=["#4C72B0", "#DD8452"], ax=axes[1, 0], bins=30)
axes[1, 0].set_title("Tenure Distribution by Churn")

# Monthly charges by churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 1],
            palette=["#4C72B0", "#DD8452"])
axes[1, 1].set_xticklabels(["No", "Yes"])
axes[1, 1].set_title("Monthly Charges by Churn")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/eda_overview.png")

# Correlation heatmap (numeric features)
plt.figure(figsize=(7, 5))
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SupportTickets", "SeniorCitizen", "Churn"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/correlation_heatmap.png")

# ----------------------------------------------------------------------
# 3. PREPROCESSING SETUP
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Preprocessing")
print("=" * 70)

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# ----------------------------------------------------------------------
# 4. TRAIN & COMPARE MODELS
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Training & comparing models")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "CV ROC-AUC (mean)": cv_scores.mean(),
    }
    results.append(metrics)
    print(f"\n{name}:")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")

results_df = pd.DataFrame(results).set_index("Model")
print("\n" + "-" * 70)
print("MODEL COMPARISON SUMMARY")
print("-" * 70)
print(results_df.round(4))
results_df.round(4).to_csv(f"{OUT_DIR}/model_comparison.csv")

# Pick best model by ROC-AUC on test set
best_model_name = results_df["ROC-AUC"].idxmax()
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# ----------------------------------------------------------------------
# 5. DETAILED EVALUATION OF BEST MODEL
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"STEP 5: Detailed evaluation of best model ({best_model_name})")
print("=" * 70)

y_pred_best = best_pipeline.predict(X_test)
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/confusion_matrix.png")

# ROC curves for all models
plt.figure(figsize=(6, 5))
for name, pipe in fitted_pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=plt.gca(), name=name)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.title("ROC Curves - Model Comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/roc_curves.png")

# Model comparison bar chart
plt.figure(figsize=(9, 5))
results_df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(kind="bar", ax=plt.gca())
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison_chart.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/model_comparison_chart.png")

# ----------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (for tree-based best model, or coefficients for LR)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Feature importance")
print("=" * 70)

feature_names = (
    numeric_cols +
    list(best_pipeline.named_steps["preprocessor"]
         .named_transformers_["cat"].named_steps["onehot"]
         .get_feature_names_out(categorical_cols))
)

classifier = best_pipeline.named_steps["classifier"]
if hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_
elif hasattr(classifier, "coef_"):
    importances = np.abs(classifier.coef_[0])
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top 15 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/feature_importance.png")
    print(fi_df.to_string(index=False))

# ----------------------------------------------------------------------
# 7. SAVE MODEL
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Saving trained model")
print("=" * 70)

model_path = f"{MODEL_DIR}/churn_model.joblib"
joblib.dump(best_pipeline, model_path)
print(f"Model saved to: {model_path}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Best model: {best_model_name} | ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
