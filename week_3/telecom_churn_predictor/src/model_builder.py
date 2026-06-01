import pandas as pd
import logging
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: Path) -> pd.DataFrame:
    logging.info(f"Loading processed data from {filepath}")
    return pd.read_csv(filepath)

def create_model_pipeline(df: pd.DataFrame) -> Pipeline:
    logging.info("Constructing the machine learning pipeline...")
    
    # Automatically identify numerical and categorical columns
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Exclude our target variable ('Churn') and numerical features to get categories
    categorical_features = [col for col in df.columns if col not in numerical_features + ['Churn']]

    # Preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])

    # Random Forest Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42
        ))
    ])
    
    return pipeline

def train_and_evaluate(df: pd.DataFrame, pipeline: Pipeline) -> Pipeline:
    logging.info("Preparing data for training...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("Training the Random Forest model...")
    pipeline.fit(X_train, y_train)

    logging.info("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("CHURN PREDICTION PERFORMANCE REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Predicted Stay, Actually Stayed): {cm[0][0]}")
    print(f"False Positives (Predicted Churn, Actually Stayed): {cm[0][1]}")
    print(f"False Negatives (Predicted Stay, Actually Churned): {cm[1][0]}")
    print(f"True Positives (Predicted Churn, Actually Churned): {cm[1][1]}")
    print("="*50 + "\n")

    return pipeline

if __name__ == "__main__":
    PROCESSED_DATA_PATH = Path("week_3/telecom_churn_predictor/data/03_processed/telco_churn_cleaned.csv")
    MODEL_OUTPUT_PATH = Path("week_3/telecom_churn_predictor/models/churn_rf_model.pkl")

    try:
        df = load_data(PROCESSED_DATA_PATH)
        ml_pipeline = create_model_pipeline(df)
        trained_model = train_and_evaluate(df, ml_pipeline)
        
        logging.info(f"Saving trained model to {MODEL_OUTPUT_PATH}")
        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_model, MODEL_OUTPUT_PATH)
        logging.info("Model saved successfully.")
        
    except FileNotFoundError:
        logging.error("Processed data not found. Run data_cleaner.py first.")