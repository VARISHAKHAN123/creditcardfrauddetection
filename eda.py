import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset ka path (ZIP ke andar creditcard.csv  hai)
csv_path = Path("C:\internproject\CreditCard_Fraud_Detection_Trends\dataset\creditcard.csv.zip")

print(f"[INFO] Loading dataset: {csv_path}")
# Dataset load karna
df = pd.read_csv(csv_path)

# Dataset ka shape (rows, columns) aur column names print karna
print("[INFO] Shape:", df.shape)
print("[INFO] Columns:", df.columns.tolist())

# Target column 'Class' ka distribution dekhna
# 0 = Non-Fraud, 1 = Fraud
print("[INFO] Class distribution:\n", df["Class"].value_counts())

# ---------- Exploratory Data Analysis (EDA) ----------

# Non-Fraud transactions ke Amount ka histogram
df[df["Class"] == 0]["Amount"].hist(bins=50, alpha=0.6, label="Non-Fraud")

# Fraud transactions ke Amount ka histogram
df[df["Class"] == 1]["Amount"].hist(bins=50, alpha=0.6, label="Fraud")

# Graph ke liye legend aur title
plt.legend()
plt.title("Amount Distribution")

# Graph ko reports folder ke andar save karna (150 dpi quality me)
plt.savefig("../reports/amount_distribution.png", dpi=150)

# Plot ko close karna (taaki memory free ho aur next plots me overlap na ho)
plt.close()

print("[OK] EDA plots saved in reports folder")

