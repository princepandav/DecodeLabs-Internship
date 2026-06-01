# 🏥 Medical Appointment "No-Show" Optimizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data Visualization](https://img.shields.io/badge/Dashboard-Streamlit-red)

##  Project Overview
Missed medical appointments (no-shows) cost healthcare systems thousands of dollars daily in lost revenue and wasted physician time. This project aims to solve this business problem by analyzing patient attendance patterns and building a machine learning model to predict the probability of a patient skipping their appointment. 

By identifying high-risk patients, clinics can optimize their scheduling systems, proactively send targeted SMS reminders, or implement double-booking strategies to minimize financial loss.

##  Features
- **Automated Data Pipeline:** Cleans messy real-world data, handles missing values, and engineers new features (like exact wait times).
- **Machine Learning Model:** Uses a highly optimized Scikit-Learn Pipeline with `LogisticRegression` (balanced class weights) and a `ColumnTransformer` to predict attendance.
- **Data Visualization:** Generates professional, presentation-ready `.png` charts to communicate insights clearly.
- **Interactive Web Dashboard:** A live Streamlit application providing interactive Plotly charts and high-level KPIs for clinic managers.

##  Project Structure
```text
medical_no_show_optimizer/
│
├── data/                   
│   ├── 01_raw/             # Original Kaggle dataset (Do not modify)
│   └── 03_processed/       # Cleaned, ML-ready dataset
│
├── src/                    
│   ├── data_cleaner.py     # Script to clean data and engineer features
│   ├── model_builder.py    # Script to train and evaluate the ML pipeline
│   ├── visualizer.py       # Script to generate static EDA charts
│   └── dashboard.py        # Streamlit interactive web dashboard
│
├── models/                 
│   └── no_show_logistic_model.pkl # Serialized Scikit-Learn model
│
├── reports/                
│   └── figures/            # Output folder for static charts (.png)
│
└── README.md