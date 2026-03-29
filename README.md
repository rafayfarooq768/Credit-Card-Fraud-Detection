# 💳 Credit Card Fraud Detection using XGBoost

## 📌 Overview
This project builds a machine learning model to detect fraudulent credit card transactions using **XGBoost**.  
The dataset is highly imbalanced, making fraud detection a challenging real-world classification problem.

---

## 📊 Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset  
- Transactions: 284,807  
- Fraud cases: ~0.17%  

### Features:
- `V1–V28`: PCA-transformed features (anonymized for privacy)  
- `Time`: seconds since first transaction  
- `Amount`: transaction amount  
- `Class`: target variable  
  - `0` → Normal  
  - `1` → Fraud  

---

## ⚙️ Approach

### 1. Data Preprocessing
- Scaled `Amount` using StandardScaler  
- Extracted time-based feature (`Hour`) from `Time`  
- Dropped unnecessary columns  

---

### 2. Handling Class Imbalance
The dataset is extremely imbalanced. To address this:
- Used `scale_pos_weight` in XGBoost  
- Ensured balanced learning between fraud and normal transactions  

---

### 3. Model
- Algorithm: **XGBoost Classifier**  
- Objective: Binary classification  

**Key parameters:**
- `n_estimators = 200`  
- `max_depth = 6`  
- `learning_rate = 0.1`  
- `scale_pos_weight` (computed from data)  

---

## 📈 Evaluation

Due to class imbalance, accuracy is not a reliable metric.  
The model is evaluated using:

- **Precision** → correctness of fraud predictions  
- **Recall** → ability to detect fraud (critical metric)  
- **F1-score** → balance between precision and recall  
- **ROC-AUC Score** → overall classification performance  

---

## 📊 Results

- ROC-AUC: ~0.99  
- Recall (Fraud Detection): ~0.86  
- Precision: ~0.88  

The model successfully detects the majority of fraudulent transactions while maintaining a low false positive rate.

---

## ⚠️ Key Insight

There is a trade-off between:
- Detecting all fraud cases (high recall)  
- Minimizing false alarms (precision)  

Lowering the classification threshold increases fraud detection but also increases false positives.

---

## 📉 Limitations

- Features are PCA-transformed → limited interpretability  
- Some fraud cases are still missed (false negatives)  
- Real-world systems often use multi-stage detection pipelines  

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

Run the notebook to:

1. Load and preprocess data  
2. Train XGBoost model  
3. Evaluate performance  
4. Visualize results  

---

## 📎 Dataset Link

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

---

## 👨‍💻 Author

Your Name Here


