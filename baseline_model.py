import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# dataset ka path
csv_path = Path("C:\internproject\CreditCard_Fraud_Detection_Trends\dataset\creditcard.csv.zip")

print(f"[INFO] Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# Add Hour feature
df["Hour"] = (df["Time"] / 3600) % 24
df = df.drop(columns=["Time"])

# Split features/target
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale Amount + Hour
scaler = StandardScaler()
X_train[["Amount", "Hour"]] = scaler.fit_transform(X_train[["Amount", "Hour"]])
X_test[["Amount", "Hour"]] = scaler.transform(X_test[["Amount", "Hour"]])

# Baseline Logistic Regression with class_weight="balanced"
model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=4))

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
