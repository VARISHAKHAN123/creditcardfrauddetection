import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA

# Dataset path
csv_path = Path("C:\internproject\CreditCard_Fraud_Detection_Trends\dataset\creditcard.csv.zip")
df = pd.read_csv(csv_path)

# Reports folder
reports = Path("../reports")
reports.mkdir(exist_ok=True)

# 1. Class Imbalance
plt.figure(figsize=(5,4))
sns.countplot(x="Class", data=df, palette="Set2")
plt.title("Class Imbalance (Fraud=1, Non-Fraud=0)")
plt.savefig(reports / "class_imbalance.png", dpi=150)
plt.close()

# 2. Amount Distribution
plt.figure(figsize=(7,5))
sns.histplot(df[df["Class"]==0]["Amount"], bins=50, color="blue", label="Non-Fraud", stat="density", alpha=0.6)
sns.histplot(df[df["Class"]==1]["Amount"], bins=50, color="red", label="Fraud", stat="density", alpha=0.6)
plt.legend()
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Density")
plt.savefig(reports / "amount_fraud_vs_nonfraud.png", dpi=150)
plt.close()

# 3. PCA Plot
X = df.drop(columns=["Class"])
y = df["Class"]

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[y==0,0], X_pca[y==0,1], label="Non-Fraud", alpha=0.4, s=2)
plt.scatter(X_pca[y==1,0], X_pca[y==1,1], label="Fraud", alpha=0.7, s=8, color="red")
plt.title("PCA Scatter Plot (Fraud vs Non-Fraud)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig(reports / "pca_fraud_vs_nonfraud.png", dpi=150)
plt.close()

print("[OK] Plots generated in reports folder")
