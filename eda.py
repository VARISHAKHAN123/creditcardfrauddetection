import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# dataset ka path
csv_path = Path("C:\internproject\CreditCard_Fraud_Detection_Trends\dataset\creditcard.csv.zip")

print(f"[INFO] Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# Shape aur info
print("[INFO] Shape:", df.shape)
print("[INFO] Columns:", df.columns.tolist())

# Class distribution
print("[INFO] Class distribution:\n", df["Class"].value_counts())

# Histogram for Amount
df[df["Class"] == 0]["Amount"].hist(bins=50, alpha=0.6, label="Non-Fraud")
df[df["Class"] == 1]["Amount"].hist(bins=50, alpha=0.6, label="Fraud")
plt.legend()
plt.title("Amount Distribution")
plt.savefig("../reports/amount_distribution.png", dpi=150)
plt.close()

print("[OK] EDA plots saved in reports folder")
