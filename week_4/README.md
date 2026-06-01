#  Spotify "Hit Song" Audio DNA Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data Visualization](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

##  Project Overview
What makes a song a global smash hit? This project moves beyond standard demographic data to deconstruct the mathematical "Audio DNA" of thousands of Spotify tracks. 

By analyzing sonic features like *Danceability*, *Acousticness*, and *Valence* (musical happiness), this project features an automated ETL pipeline, advanced feature engineering, a highly optimized gradient boosting predictive model, and a fully interactive web dashboard. It is designed to act as a data-driven A&R (Artists and Repertoire) tool to predict track popularity and uncover niche acoustic trends.

## 🛠️ Key Features
- **Automated Data Pipeline:** Cleans messy Spotify dumps, handles duplicates, standardizes track lengths, and encodes explicit content flags.
- **Advanced Feature Engineering:** Extracts "hidden" data patterns by calculating custom metrics like the *Loud & Fast Factor*, *Acoustic Purity*, and the *Summer Hit Vibe Score*.
- **Highly Optimized Machine Learning:** Utilizes Scikit-Learn's `HistGradientBoostingRegressor` (an ultra-fast, histogram-based algorithm) with cross-validated hyperparameter tuning (`RandomizedSearchCV`) to predict track popularity with high accuracy.
- **Corporate Visual Report Suite:** Automatically generates presentation-ready static charts, including an advanced Audio Profile Radar (Spider) Chart comparing Smash Hits to Unpopular tracks.
- **Dynamic Web Dashboard:** A live Streamlit application with interactive Plotly visuals, allowing users to filter global tracks by genre, popularity, and explicit content in real-time.

##  Project Structure
```text
spotify_hit_analyzer/
│
├── data/                   
│   ├── 01_raw/             # Original Kaggle dataset (Do not modify)
│   └── 03_processed/       # Cleaned, ML-ready dataset
│
├── src/                    
│   ├── data_cleaner.py     # ETL script to clean data and format features
│   ├── model_builder.py    # Feature engineering, hyperparameter tuning, and ML training
│   ├── visualizer.py       # Script to generate static EDA charts for PDF reports
│   └── dashboard.py        # Streamlit interactive web dashboard
│
├── models/                 
│   └── song_popularity_model.pkl # Serialized, tuned Gradient Boosting pipeline
│
├── reports/                
│   ├── figures/            # Output folder for static charts (.png)
│   └── final_presentation.pdf # Manually generated executive summary
│
└── README.md