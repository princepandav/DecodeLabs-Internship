# 🚀 Decodelabs Data Science Internship Portfolio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![Data Visualization](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📌 About This Repository
This repository contains the complete collection of projects developed during the **Decodelabs Data Science Internship**. Over the course of four weeks, this internship focused on practical, hands-on applications of data science, transitioning from raw data handling to building production-ready predictive models and interactive business dashboards.

Each week explores a distinct industry domain, applying advanced machine learning algorithms to solve real-world problems.

## 📂 Repository Structure & Project Directory

```text
DECODELABS INTERNSHIP/
│
├── week_1/ # Algorithmic Trading & Market Prediction
├── week_2/ # Healthcare Operations Optimization
├── week_3/ # Telecom Customer Retention Analytics
├── week_4/ # Entertainment & Audio Feature Analysis
└── README.md


📈 Week 1: Algorithmic Trading & Risk-Reward Modeling
Domain: Quantitative Finance

Overview: A predictive financial model designed to analyze market data (e.g., Nifty 50 F&O) and identify high-probability trading setups.

Core Technology: Built using an XGBoost machine learning architecture to train on historical price action and volume data.

Business Value: The model predicts future market direction while calculating precise risk-to-reward ratios, enabling automated, data-driven entry and exit strategies rather than relying on emotional trading.

🏥 Week 2: Medical Appointment "No-Show" Optimizer
Domain: Healthcare Administration

Overview: An end-to-end data pipeline and classification system built to predict whether a patient will miss their scheduled clinic appointment.

Core Technology: Utilizes a highly optimized Logistic Regression pipeline with a ColumnTransformer to handle both numerical and categorical patient data.

Business Value: Features an interactive Streamlit dashboard that allows clinic managers to identify high-risk patients in real-time, enabling targeted SMS reminders and double-booking strategies to prevent revenue loss.

📡 Week 3: Telecom Customer Churn Predictor
Domain: Business Operations & Marketing

Overview: A predictive analytics tool designed to flag active subscribers who exhibit behavioral patterns suggesting they are about to cancel their service.

Core Technology: Powered by a hyperparameter-tuned Random Forest Classifier optimized via Cross-Validated Grid Search (RandomizedSearchCV).

Business Value: Uncovers the financial impact of contract types and monthly charges, providing marketing teams with a dynamic dashboard to segment at-risk users and deploy retention incentives before the customer churns.

🎵 Week 4: Spotify "Hit Song" Audio DNA Analyzer
Domain: Music Industry & Entertainment

Overview: An advanced analytical tool that moves beyond demographics to deconstruct the mathematical "Audio DNA" (Danceability, Valence, Energy) of global hit songs.

Core Technology: Features complex feature engineering and utilizes Scikit-Learn's lightning-fast HistGradientBoostingRegressor to predict track popularity.

Business Value: Includes a highly interactive web dashboard with an Audio Profile Radar Chart, allowing A&R executives to instantly visualize acoustic trends and predict if a new track fits the mathematical formula of a global smash hit.

🛠️ Core Tech Stack
Across the four weeks, the following libraries and frameworks were utilized to build these solutions:

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn, xgboost, joblib

Data Visualization: matplotlib, seaborn, plotly

Web Deployment: streamlit

🚀 How to Navigate
Each weekly folder acts as a standalone project with its own isolated environment. Navigate to any specific week's folder to find a dedicated README.md containing precise instructions on how to install dependencies, run the ETL pipelines, and launch the interactive dashboards.

👨‍💻 Author
Developed by a dedicated data science intern as part of the Decodelabs curriculum, demonstrating the ability to take raw datasets and transform them into actionable, production-ready business intelligence tools.