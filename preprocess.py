import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("[INFO] Train shape:", X_train.shape, " Test shape:", X_test.shape)

# Scale Amount + Hour
scaler = StandardScaler()
X_train[["Amount", "Hour"]] = scaler.fit_transform(X_train[["Amount", "Hour"]])
X_test[["Amount", "Hour"]] = scaler.transform(X_test[["Amount", "Hour"]])

print("[OK] Preprocessing done")
