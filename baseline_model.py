import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# Dataset ka path (yaha ZIP file me CSV pada hai)
csv_path = Path("C:\internproject\CreditCard_Fraud_Detection_Trends\dataset\creditcard.csv.zip")

print(f"[INFO] Loading dataset: {csv_path}")
# Dataset load karna
df = pd.read_csv(csv_path)

# 'Time' feature ko Hours me convert karke naya column 'Hour' add karna
df["Hour"] = (df["Time"] / 3600) % 24
# Ab 'Time' column ki zarurat nahi hai, isliye drop kar diya
df = df.drop(columns=["Time"])

# Features (X) aur Target (y) alag karna
X = df.drop(columns=["Class"])  # Independent variables
y = df["Class"]                # Dependent variable (fraud = 1, non-fraud = 0)

# Data ko Train aur Test set me split karna (80% train, 20% test)
# stratify=y use kiya gaya hai taaki fraud aur non-fraud ka proportion maintain ho
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 'Amount' aur 'Hour' columns ko scale karna (Normalization)
# Kyunki unki values bahut different range me hoti hain
scaler = StandardScaler()
X_train[["Amount", "Hour"]] = scaler.fit_transform(X_train[["Amount", "Hour"]])
X_test[["Amount", "Hour"]] = scaler.transform(X_test[["Amount", "Hour"]])

# Logistic Regression model banaya with class_weight="balanced"
# Iska matlab minority class (fraud cases) ko extra importance milegi
model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)  # Model ko train karna

# Predictions lena (class aur probabilities dono)
y_pred = model.predict(X_test)                  # Final class prediction (0/1)
y_prob = model.predict_proba(X_test)[:, 1]      # Fraud hone ki probability

# Evaluation metrics print karna
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=4))  # Precision, Recall, F1-Score

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))  # True Negative, False Positive, False Negative, True Positive

# ROC-AUC aur PR-AUC (imbalanced data ke liye important metrics)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
