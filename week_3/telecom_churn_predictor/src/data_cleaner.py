import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: Path) -> pd.DataFrame:
    logging.info(f"Loading raw data from {filepath}")
    return pd.read_csv(filepath)

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting data cleaning process...")
    df_clean = df.copy()

    # 1. Fix the 'TotalCharges' hidden string issue
    # errors='coerce' forces any blank spaces to become NaN (Not a Number)
    logging.info("Converting TotalCharges to numeric format...")
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    # 2. Handle Missing Values
    # Since it's only about 11 rows (brand new customers with 0 tenure), we can drop them
    initial_rows = len(df_clean)
    df_clean.dropna(subset=['TotalCharges'], inplace=True)
    logging.info(f"Dropped {initial_rows - len(df_clean)} rows with missing TotalCharges.")

    # 3. Convert Target Variable to Binary
    # Machine learning models need 1s and 0s, not 'Yes' and 'No'
    logging.info("Encoding target variable 'Churn' to binary...")
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

    # 4. Drop non-predictive columns
    # Customer IDs are unique to everyone and will confuse the ML model
    df_clean.drop(columns=['customerID'], inplace=True, errors='ignore')

    logging.info("Data cleaning completed successfully.")
    return df_clean

def save_data(df: pd.DataFrame, output_path: Path):
    logging.info(f"Saving processed data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    RAW_DATA_PATH = Path("week_3/telecom_churn_predictor/data/01_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    PROCESSED_DATA_PATH = Path("week_3/telecom_churn_predictor/data/03_processed/telco_churn_cleaned.csv")

    try:
        raw_df = load_data(RAW_DATA_PATH)
        cleaned_df = clean_and_preprocess(raw_df)
        save_data(cleaned_df, PROCESSED_DATA_PATH)
        logging.info(f"Dataset ready. Shape: {cleaned_df.shape}")
    except FileNotFoundError:
        logging.error(f"Dataset not found at {RAW_DATA_PATH}. Please check the file name.")