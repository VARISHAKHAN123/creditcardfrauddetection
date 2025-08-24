# creditcardfrauddetection
Credit Card Fraud Detection Trends
# Credit Card Fraud Detection Trends

## 📌 Introduction
This project analyzes the **Kaggle Credit Card Fraud dataset** to explore fraud detection patterns.  
The focus is on **data imbalance visualization, transaction amount patterns, and PCA plots** to study separation between fraud and non-fraud transactions.  

- Total Transactions: **284,807**
- Fraud Cases: **492 (0.17%)**
- Non-Fraud Cases: **284,315**

---

## 🎯 Goals
1. Visualize **class imbalance** between fraud and non-fraud transactions  
2. Study **transaction amount patterns** (fraud vs non-fraud)  
3. Use **PCA (Principal Component Analysis)** to visualize fraud vs non-fraud separation  

---

## 📊 Exploratory Data Analysis

### 1. Class Imbalance
Fraud cases are extremely rare compared to non-fraud.  

![Class Imbalance](reports/class_imbalance.png)

---

### 2. Amount Distribution
Fraud transactions are generally lower in amount compared to non-fraud.  

![Amount Distribution](reports/amount_fraud_vs_nonfraud.png)

---

### 3. PCA Visualization
PCA scatter plot shows how fraud and non-fraud transactions are distributed in lower dimensions.  
Fraud cases are scattered and overlap with normal transactions, making detection challenging.  

![PCA Plot](reports/pca_fraud_vs_nonfraud.png)

---

## ✅ Conclusion
- Dataset is **highly imbalanced** (only 0.17% fraud).  
- Fraud transactions often differ in **amount patterns**.  
- PCA shows that fraud is not clearly separable, which makes detection a **challenging ML problem**.  

This analysis sets the foundation for building advanced fraud detection models in the future.  
