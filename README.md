# 🚀 Customer Retention Prediction

Predict whether a customer will churn using historical data. This project implements a **complete machine learning pipeline**, from data exploration to advanced model tuning using Optuna, and evaluates performance across multiple models.  

---

## 📂 Project Structure
```bash
Customer-Retention-Prediction/
│
├── Customer Retention Prediction.ipynb # Baseline models & initial performance
├── Performance Comparison With Optuna.ipynb # Optuna hyperparameter tuning & comparison
├── dataset.csv # Customer dataset
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```


---

## 📝 Project Overview

Customer churn is a critical issue for businesses, impacting revenue and growth. The goal of this project is to predict customer turnover using **machine learning and deep learning models**.  

Key steps include:  

### 1️⃣ Exploratory Data Analysis (EDA)
- Identify key features  
- Handle missing values and duplicates  
- Visualize trends and patterns  
- Identify top features affecting churn  

### 2️⃣ Data Preprocessing
- Encode categorical features  
- Scale/normalize numerical features  
- Split dataset into train, validation, and test sets  
- Handle class imbalance  

### 3️⃣ Feature Engineering
- Create meaningful features to improve predictive performance  

### 4️⃣ Model Design
- **Baseline models**: RandomForest, XGBoost, LightGBM, Logistic Regression, MLP, TabTransformer  
- **Advanced models**: Multi-layer Perceptron (MLP) tuned with Optuna  
- **Hyperparameter tuning** with Optuna  

### 5️⃣ Performance Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- Visualizations: Confusion Matrix, ROC Curve  
- Compare baseline vs tuned models  

---

## 📊 Baseline Model Performance

| Model            | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|-----------------|---------|-----------|--------|----------|---------|
| TabTransformer   | 0.540   | 0.546     | 0.747  | 0.631    | 0.538   |
| MLP              | 0.540   | 0.557     | 0.620  | 0.587    | 0.524   |
| RandomForest     | 0.427   | 0.462     | 0.532  | 0.494    | 0.462   |
| Logistic         | 0.440   | 0.466     | 0.430  | 0.447    | 0.438   |
| XGBoost          | 0.427   | 0.460     | 0.506  | 0.482    | 0.430   |
| LightGBM         | 0.447   | 0.474     | 0.456  | 0.465    | 0.425   |

---

## 📈 Optuna-Tuned MLP Results

After hyperparameter tuning with **Optuna**, the MLP performance improved:

| Metric      | Value    |
|------------|----------|
| Accuracy   | 0.5350   |
| Precision  | 0.5500   |
| Recall     | 0.6286   |
| F1-Score   | 0.5867   |
| ROC-AUC    | 0.5301   |

**Best Hyperparameters:**
```json
{
  "n_layers": 3,
  "n_units_l0": 227,
  "n_units_l1": 235,
  "n_units_l2": 118,
  "learning_rate_init": 1.78e-05,
  "alpha": 1.68e-05
}
```

## 📊 Key Insights

- **TabTransformer** and **MLP** are the strongest baseline models, achieving similar accuracy (~**54%**).  
- **Tree-based models** such as RandomForest, XGBoost, and LightGBM perform moderately (~**42–45%** accuracy).  
- **Optuna hyperparameter tuning** improves the MLP’s balance between precision and recall, slightly boosting **F1-score** and **ROC-AUC**.  
- Key churn predictors include:
  - 🕒 Customer tenure  
  - 💰 Total spend  
  - 🔁 Purchase frequency  
  - 🎯 Promotion response  

⚙️ Setup Instructions

Clone the repository:
```bash
git clone https://github.com/yourusername/Customer-Retention-Prediction.git
cd Customer-Retention-Prediction

```
Install dependencies:
```bash
pip install -r requirements.txt

```
Open notebooks in Jupyter Notebook / Google Colab:
```bash
jupyter notebook
```

## ⚙️ Execution Order

To reproduce the full pipeline, run the notebooks sequentially as follows:

1. **Customer Retention Prediction.ipynb**  
   → Performs data preprocessing, feature engineering, and baseline model training (RandomForest, XGBoost, LightGBM, MLP, TabTransformer, etc.).

2. **Performance Comparison With Optuna.ipynb**  
   → Applies Optuna-based hyperparameter tuning and compares optimized model performance across multiple metrics.

> 💡 **Tip:**  
> Run the notebooks in this order to ensure all intermediate files and model artifacts are properly generated before evaluation.


