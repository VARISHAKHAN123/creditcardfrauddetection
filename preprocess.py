import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset ka path (zip file me creditcard.csv pada hai)
csv_path = Path("C:\internproject\CreditCard_Fraud_Detection_Trends\dataset\creditcard.csv.zip")

print(f"[INFO] Loading dataset: {csv_path}")
# Dataset load karna
df = pd.read_csv(csv_path)

# Naya feature "Hour" create karna
# 'Time' column transaction ka time batata hai (seconds me)
# Use 3600 (seconds in hour) se divide karke hours me convert kiya
# Phir % 24 liya taaki value 0-23 hours ke beech me ho
df["Hour"] = (df["Time"] / 3600) % 24

# 'Time' column ki ab zarurat nahi, isliye drop kar diya
df = df.drop(columns=["Time"])

# Features (X) aur Target (y) alag karna
X = df.drop(columns=["Class"])  # saare independent variables
y = df["Class"]                 # dependent variable (fraud = 1, non-fraud = 0)

# Dataset ko train aur test set me split karna (80% train, 20% test)
# stratify=y ka matlab hai fraud aur non-fraud ka ratio train/test dono me same rahega
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("[INFO] Train shape:", X_train.shape, " Test shape:", X_test.shape)

# Amount aur Hour columns ka scale karna
# Kyunki unki values kaafi different range me hoti hain
scaler = StandardScaler()
X_train[["Amount", "Hour"]] = scaler.fit_transform(X_train[["Amount", "Hour"]])  # Train data se scaler fit aur transform
X_test[["Amount", "Hour"]] = scaler.transform(X_test[["Amount", "Hour"]])        # Test data me same scaler apply

print("[OK] Preprocessing done")
