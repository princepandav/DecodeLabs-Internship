#  Telecom Customer Churn Prediction & Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data Visualization](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

##  Project Overview
Customer acquisition is vastly more expensive than customer retention. This project addresses a critical business problem for the telecommunications industry: predicting which customers are at a high risk of canceling their subscriptions (churning). 

By leveraging historical customer data, this project features an automated data pipeline, an advanced machine learning classification model, and an interactive web dashboard. These tools empower business executives and marketing teams to identify high-risk segments and proactively offer retention incentives before a customer leaves.

##  Key Features
- **Automated Data Cleaning Pipeline:** Seamlessly handles Kaggle's notoriously messy Telco dataset, fixing hidden string errors, dropping anomalies, and encoding categorical variables.
- **Advanced Machine Learning:** Utilizes a highly optimized Scikit-Learn `RandomForestClassifier`. The model undergoes rigorous hyperparameter tuning via Cross-Validated Grid Search (`RandomizedSearchCV`) to ensure maximum predictive accuracy.
- **Automated Reporting Suite:** Generates presentation-ready, high-resolution `.png` visualizations that clearly explain the exact factors driving customer churn.
- **Dynamic Web Dashboard:** A live, interactive Streamlit application that allows users to slice and filter data on the fly to uncover niche customer behavioral trends.

##  Project Structure
```text
telecom_churn_predictor/
│
├── data/                   
│   ├── 01_raw/             # Original Kaggle dataset (Do not modify)
│   └── 03_processed/       # Cleaned, ML-ready dataset
│
├── src/                    
│   ├── data_cleaner.py     # ETL script to clean data and handle missing values
│   ├── model_builder.py    # Script for hyperparameter tuning and model training
│   ├── visualizer.py       # Script to generate static EDA charts for PDF reports
│   └── dashboard.py        # Streamlit interactive web dashboard
│
├── models/                 
│   └── churn_rf_optimized_model.pkl # Serialized, tuned Random Forest pipeline
│
├── reports/                
│   ├── figures/            # Output folder for static charts (.png)
│   └── final_presentation.pdf # Manually generated executive summary
│
└── README.md