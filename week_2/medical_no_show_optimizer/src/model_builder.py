import pandas as pd
import logging
import joblib
from pathlib import Path

# Scikit-Learn components
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the preprocessed CSV dataset."""
    logging.info(f"Loading processed data from {filepath}")
    return pd.read_csv(filepath)

def create_model_pipeline() -> Pipeline:
    """
    Constructs a highly optimized Scikit-Learn Pipeline.
    Handles numerical scaling and categorical encoding automatically.
    """
    logging.info("Constructing the machine learning pipeline...")
    
    # Define which columns need which preprocessing
    numerical_features = ['Age', 'AwaitingTime']
    categorical_features = ['Gender', 'Neighbourhood']
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' 
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    return pipeline

def train_and_evaluate_model(df: pd.DataFrame, pipeline: Pipeline) -> Pipeline:
    """Trains the model and evaluates its performance."""
    logging.info("Preparing data for training...")

    X = df.drop(columns=['No_Show'])
    y = df['No_Show']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(f"Training set shape: {X_train.shape} | Test set shape: {X_test.shape}")
    logging.info("Training the Logistic Regression model (this may take a moment)...")
    
    pipeline.fit(X_train, y_train)

    logging.info("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("MODEL PERFORMANCE REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Showed Up (0)', 'No-Show (1)']))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}  | True Positives: {cm[1][1]}")
    print("="*50 + "\n")

    return pipeline

def save_model(model: Pipeline, output_path: Path):
    """Serializes and saves the trained model to the disk."""
    logging.info(f"Saving trained model to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logging.info("Model saved successfully. Pipeline execution complete.")

if __name__ == "__main__":

    PROCESSED_DATA_PATH = Path("week_2/medical_no_show_optimizer/data/03_processed/medical_no_shows_cleaned.csv")
    MODEL_OUTPUT_PATH = Path("week_2/medical_no_show_optimizer/models/no_show_logistic_model.pkl")

    try:
        df = load_data(PROCESSED_DATA_PATH)
        
        ml_pipeline = create_model_pipeline()
        trained_model = train_and_evaluate_model(df, ml_pipeline)

        save_model(trained_model, MODEL_OUTPUT_PATH)
        
    except FileNotFoundError:
        logging.error(f"Could not find the dataset at {PROCESSED_DATA_PATH}. Run data_cleaner.py first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during model building: {e}")