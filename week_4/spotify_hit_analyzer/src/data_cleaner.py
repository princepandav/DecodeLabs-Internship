import pandas as pd
import logging
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the raw CSV dataset into a Pandas DataFrame."""
    logging.info(f"Loading raw data from {filepath}")
    return pd.read_csv(filepath)

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Applies data cleaning and formatting to the Spotify dataset."""
    logging.info("Starting data cleaning process...")
    df_clean = df.copy()

    # 1. Drop the unnamed index column
    # Often exported from pandas as 'Unnamed: 0', representing that stray comma in the raw CSV
    if 'Unnamed: 0' in df_clean.columns:
        logging.info("Dropping 'Unnamed: 0' index column...")
        df_clean.drop(columns=['Unnamed: 0'], inplace=True)

    # 2. Handle missing critical text values
    initial_rows = len(df_clean)
    df_clean.dropna(subset=['track_name', 'artists', 'album_name'], inplace=True)
    if initial_rows - len(df_clean) > 0:
        logging.info(f"Dropped {initial_rows - len(df_clean)} rows with missing track or artist names.")

    # 3. Remove duplicate tracks
    # Spotify data often contains duplicates if a song is on multiple albums or playlists
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(subset=['track_name', 'artists'], keep='first', inplace=True)
    logging.info(f"Dropped {initial_rows - len(df_clean)} duplicate tracks to prevent model bias.")

    # 4. Feature Engineering: Convert 'duration_ms' to 'duration_min'
    # Machine learning handles standard minutes better than massive millisecond numbers
    if 'duration_ms' in df_clean.columns:
        logging.info("Converting 'duration_ms' to standard minutes...")
        df_clean['duration_min'] = df_clean['duration_ms'] / 60000.0
        df_clean.drop(columns=['duration_ms'], inplace=True)

    # 5. Convert Boolean 'explicit' column to Binary (1/0)
    if 'explicit' in df_clean.columns:
        logging.info("Converting 'explicit' flag to binary...")
        df_clean['explicit'] = df_clean['explicit'].astype(int)

    logging.info("Data cleaning completed successfully.")
    return df_clean

def save_data(df: pd.DataFrame, output_path: Path):
    """Saves the cleaned DataFrame to the specified output directory."""
    logging.info(f"Saving processed data to {output_path}")
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save without the index column
    df.to_csv(output_path, index=False)
    logging.info("Data saved successfully.")

if __name__ == "__main__":
    # Define file paths
    # Note: Ensure your downloaded file name matches the RAW_DATA_PATH below
    RAW_DATA_PATH = Path("week_4/spotify_hit_analyzer/data/01_raw/spotify_dataset.csv") 
    PROCESSED_DATA_PATH = Path("week_4/spotify_hit_analyzer/data/03_processed/spotify_cleaned.csv")

    try:
        # Run the ETL (Extract, Transform, Load) pipeline
        raw_dataframe = load_data(RAW_DATA_PATH)
        cleaned_dataframe = clean_and_preprocess(raw_dataframe)
        save_data(cleaned_dataframe, PROCESSED_DATA_PATH)
        
        logging.info(f"Pipeline finished. Final dataset shape: {cleaned_dataframe.shape}")
        
    except FileNotFoundError:
        logging.error(f"Could not find the dataset at {RAW_DATA_PATH}. Please ensure the file exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")