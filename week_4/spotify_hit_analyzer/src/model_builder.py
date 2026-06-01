import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

# Scikit-Learn modules
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the preprocessed dataset."""
    logging.info(f"Loading cleaned data from {filepath}")
    return pd.read_csv(filepath)

def engineer_hidden_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds 'hidden data' patterns by mathematically combining audio features 
    to create powerful new predictive columns.
    """
    logging.info("Engineering hidden musical features...")
    df_engineered = df.copy()

    # Pattern 1: The "Loud & Fast" Factor (High energy + Loudness + Tempo)
    # Loudness is usually negative, so we add 60 to baseline it before multiplying
    df_engineered['loud_fast_factor'] = df_engineered['energy'] * (df_engineered['loudness'] + 60) * df_engineered['tempo']

    # Pattern 2: The "Pure Acoustic" Index
    # Songs with high acousticness and low energy have a distinct demographic
    df_engineered['acoustic_purity'] = df_engineered['acousticness'] / (df_engineered['energy'] + 0.01)

    # Pattern 3: The "Vibe" Score (Positivity vs. Danceability)
    # Valence is musical happiness. High valence + high danceability = Summer Hit
    df_engineered['vibe_score'] = df_engineered['valence'] * df_engineered['danceability']

    logging.info("Feature engineering complete. Hidden patterns exposed.")
    return df_engineered

def create_model_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Constructs a highly optimized preprocessing and modeling pipeline.
    """
    logging.info("Constructing the ML pipeline...")
    
    # We drop ID/Name columns (they have no mathematical value) and our Target (popularity)
    exclude_cols = ['track_id', 'artists', 'album_name', 'track_name', 'popularity']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Identify numerical vs categorical features dynamically
    categorical_features = ['key', 'mode', 'track_genre']
    numerical_features = [col for col in feature_cols if col not in categorical_features]

    # Preprocessor handles different data types simultaneously
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # HistGradientBoosting is Scikit-Learn's fastest, most optimized algorithm for large datasets
    # It natively groups data into histograms, making it lightning-fast and highly accurate
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=42))
    ])
    
    return pipeline, feature_cols

def optimize_and_train(df: pd.DataFrame, pipeline: Pipeline, feature_cols: list) -> Pipeline:
    """
    Splits data, performs cross-validated hyperparameter tuning, 
    and evaluates the model's accuracy.
    """
    logging.info("Splitting data into training and testing sets...")
    X = df[feature_cols]
    y = df['popularity'] # Target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logging.info("Starting advanced hyperparameter tuning (Grid Search)...")
    
    # Define the parameter grid to test
    param_grid = {
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_iter': [100, 200, 300], # Number of trees
        'regressor__max_depth': [5, 10, None],
        'regressor__l2_regularization': [0.0, 0.1, 1.0] # Prevents overfitting
    }

    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )

    # Train the model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    logging.info(f"Best Hyperparameters Found: {search.best_params_}")
    logging.info("Evaluating optimized model on unseen test data...")
    
    # Generate predictions
    y_pred = best_model.predict(X_test)

    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*60)
    print("🎵 SPOTIFY POPULARITY PREDICTION REPORT")
    print("="*60)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f" -> (On average, our prediction is off by just {mae:.2f} popularity points)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-Squared (Accuracy Score): {r2:.4f}")
    print("="*60 + "\n")

    return best_model

def save_model(model: Pipeline, output_path: Path):
    """Serializes and saves the fully optimized model to disk."""
    logging.info(f"Saving optimized model to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logging.info("Model saved successfully. Ready for production dashboard.")

if __name__ == "__main__":
    # Define file paths
    PROCESSED_DATA_PATH = Path("week_4/spotify_hit_analyzer/data/03_processed/spotify_cleaned.csv")
    MODEL_OUTPUT_PATH = Path("week_4/spotify_hit_analyzer/models/song_popularity_model.pkl")

    try:
        # Run the full ML Pipeline
        cleaned_df = load_data(PROCESSED_DATA_PATH)
        
        # 1. Feature Engineering (Find hidden data patterns)
        engineered_df = engineer_hidden_features(cleaned_df)
        
        # 2. Build Pipeline
        base_pipeline, features = create_model_pipeline(engineered_df)
        
        # 3. Train & Optimize
        optimized_model = optimize_and_train(engineered_df, base_pipeline, features)
        
        # 4. Save to Disk
        save_model(optimized_model, MODEL_OUTPUT_PATH)
        
    except FileNotFoundError:
        logging.error(f"Dataset not found at {PROCESSED_DATA_PATH}. Run data_cleaner.py first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during model training: {e}")