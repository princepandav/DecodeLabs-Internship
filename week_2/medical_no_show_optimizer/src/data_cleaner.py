import pandas as pd
import logging
from pathlib import Path

# Configure logging for production-level tracking
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the raw CSV dataset."""
    logging.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all cleaning and feature engineering steps to the dataframe."""
    logging.info("Starting data cleaning process...")
    df_clean = df.copy()

    # 1. Standardize column names and fix Kaggle typos
    df_clean.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'No-show': 'No_Show'
    }, inplace=True)

    logging.info("Converting datetime columns...")

    df_clean['ScheduledDay'] = pd.to_datetime(df_clean['ScheduledDay']).dt.tz_localize(None)
    df_clean['AppointmentDay'] = pd.to_datetime(df_clean['AppointmentDay']).dt.tz_localize(None)
    df_clean['ScheduledDate'] = df_clean['ScheduledDay'].dt.normalize()
    df_clean['AwaitingTime'] = (df_clean['AppointmentDay'] - df_clean['ScheduledDate']).dt.days

    logging.info("Removing anomalies...")
    initial_rows = len(df_clean)
    
    df_clean = df_clean[df_clean['AwaitingTime'] >= 0]
    
    df_clean = df_clean[df_clean['Age'] >= 0]
    
    logging.info(f"Dropped {initial_rows - len(df_clean)} anomalous rows.")


    logging.info("Encoding target variable to binary...")
    df_clean['No_Show'] = df_clean['No_Show'].map({'Yes': 1, 'No': 0})

    df_clean.drop(
        columns=['PatientId', 'AppointmentID', 'ScheduledDate', 'ScheduledDay', 'AppointmentDay'], 
        inplace=True, 
        errors='ignore'
    )

    logging.info("Data cleaning completed successfully.")
    return df_clean

def save_data(df: pd.DataFrame, output_path: Path):
    """Saves the cleaned dataframe to the specified output path."""
    logging.info(f"Saving processed data to {output_path}")
    # Ensure the output directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":

    RAW_DATA_PATH = Path("week_2/medical_no_show_optimizer/data/01_raw/KaggleV2-May-2016.csv")
    PROCESSED_DATA_PATH = Path("week_2/medical_no_show_optimizer/data/03_processed/medical_no_shows_cleaned.csv")

    try:
        # Execute the pipeline
        raw_dataframe = load_data(RAW_DATA_PATH)
        cleaned_dataframe = clean_and_preprocess(raw_dataframe)
        save_data(cleaned_dataframe, PROCESSED_DATA_PATH)
        
        logging.info(f"Final dataset ready for modeling. Shape: {cleaned_dataframe.shape}")
        
    except FileNotFoundError:
        logging.error(f"Could not find the dataset at {RAW_DATA_PATH}. Please verify the file exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")