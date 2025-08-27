# ❤️ Heart Failure Mortality Prediction Using Machine Learning

Welcome to our healthcare analytics project, where data meets compassion. This repository presents a comprehensive machine learning pipeline designed to predict mortality in heart failure patients using clinical data. Built with Python and powered by rigorous preprocessing, model tuning, and ethical foresight, this project aims to support clinicians in making timely, life-saving decisions.

---

## 📌 Project Objective

Heart failure (HF) affects over 10% of individuals aged 70+, with a 5-year mortality rate. Traditional risk models often lack precision and adaptability. Our goals:

- Predict mortality outcomes using clinical features
- Evaluate six machine learning models
- Address data limitations through augmentation and balancing
- Recommend strategies for clinical integration

---

## 🧠 Dataset Overview

- **Source**: UCI Heart Failure Clinical Records Dataset  
- **Size**: 299 patient records  
- **Features**: 13 clinical attributes including age, serum creatinine, ejection fraction, and DEATH_EVENT (target)  
- **Challenge**: Class imbalance (32% mortality)

---

## 🧪 Data Preprocessing

We employed a robust preprocessing pipeline:

- ✅ No missing values
- 📊 Feature scaling with `StandardScaler`
- ⚖️ Class balancing using **SMOTE**
- 🔁 Data augmentation with Gaussian noise
- 📈 EDA with histograms, KDE plots, and correlation matrices

### 🔍 Key Predictors Identified

| Feature            | Correlation with Mortality |
|--------------------|----------------------------|
| Time               | -0.53                      |
| Serum Creatinine   | +0.29                      |
| Ejection Fraction  | -0.27                      |
| Age                | +0.25                      |

---

## ⚙️ Methodology

Six models were trained and evaluated using 10-fold cross-validation and `GridSearchCV`:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Bagging Classifier
- LightGBM
- XGBoost

### 🧮 Model Performance Summary

| Model               | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|--------------------|----------|---------|-----------|--------|----------|
| Logistic Regression| 0.78     | 0.89    | 0.69      | 0.58   | 0.63     |
| SVM                | 0.75     | 0.82    | 0.61      | 0.58   | 0.59     |
| Decision Tree      | 0.80     | 0.77    | 0.68      | 0.68   | 0.68     |
| Bagging            | 0.83     | 0.87    | 0.91      | 0.53   | 0.67     |
| LightGBM           | 0.82     | 0.86    | 0.75      | 0.63   | 0.69     |
| **XGBoost**        | **0.85** | **0.90**| 0.81      | 0.68   | 0.74     |

---

## 🏥 Clinical Implications

- **Risk Stratification**: Early identification of high-risk patients
- **Mortality Reduction**: XGBoost could reduce mortality by 15–20%
- **Decision Support**: Integration into clinical systems for personalized care

---

## 🔐 Ethical Considerations

- 🔒 HIPAA-compliant data handling
- ⚖️ Bias mitigation through fairness audits
- 🧾 Transparent model reporting and informed consent protocols

---

## 🔮 Future Work

- Validate models on external datasets like **MIMIC-III**
- Integrate real-time data from wearables and EHRs
- Explore **LSTM** networks for temporal modeling

---

## 🧑‍💻 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, XGBoost, LightGBM)
- Matplotlib & Seaborn for visualization
- SMOTE for class balancing
- GridSearchCV for hyperparameter tuning

---

## 🙌 Acknowledgements

This project was developed by **Group 5** of the **DAB 304 Healthcare Analytics** course at **St. Clair College**, Windsor, Canada.  
Special thanks to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) for the dataset and to all team members for their contributions.

---

## 📂 Repository Structure

```bash
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── notebooks/
│   └── heart_failure_analysis.ipynb
├── models/
│   └── trained_models.pkl
├── results/
│   └── performance_metrics.csv
├── README.md
└── requirements.txt
