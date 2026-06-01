import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from math import pi

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set corporate styling for the visual report
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300 # High resolution for PDF exports

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the preprocessed dataset for visualization."""
    logging.info(f"Loading cleaned data from {filepath}")
    return pd.read_csv(filepath)

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Generates a heatmap to show how audio features correlate with popularity."""
    logging.info("Generating Audio Feature Correlation Heatmap...")
    
    plt.figure(figsize=(10, 8))
    # Select only the relevant numerical audio features
    features = ['popularity', 'danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    corr_matrix = df[features].corr()
    
    # Custom color map from red (negative) to green (positive)
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Correlation of Audio Features with Track Popularity', fontsize=16, fontweight='bold')
    
    save_path = output_dir / "01_audio_correlation_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_energy_vs_acousticness(df: pd.DataFrame, output_dir: Path):
    """Creates a scatter plot showing the inverse relationship between energy and acoustics."""
    logging.info("Generating Energy vs. Acousticness Scatter Plot...")
    
    plt.figure()
    # Hexbin plot is highly optimized for thousands of overlapping data points
    plt.hexbin(df['acousticness'], df['energy'], gridsize=50, cmap='Blues', mincnt=1)
    cb = plt.colorbar(label='Number of Tracks')
    
    plt.title('The Acoustic Divide: Energy vs. Acousticness', fontsize=16, fontweight='bold')
    plt.xlabel('Acousticness (0 = Electronic, 1 = Unplugged)')
    plt.ylabel('Energy Intensity')
    
    save_path = output_dir / "02_energy_vs_acousticness.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_vibe_vs_danceability(df: pd.DataFrame, output_dir: Path):
    """Visualizes the 'Summer Hit' zone combining Valence (Happiness) and Danceability."""
    logging.info("Generating Vibe vs. Danceability KDE Plot...")
    
    plt.figure()
    # Filter for highly popular songs (>80) to see what makes a "Hit"
    hits = df[df['popularity'] >= 80]
    
    sns.kdeplot(x=hits['valence'], y=hits['danceability'], cmap="magma", fill=True, thresh=0.05)
    
    plt.title('The "Hit Song" Zone: Happiness vs. Danceability (Top Tracks)', fontsize=16, fontweight='bold')
    plt.xlabel('Valence (Musical Positivity/Happiness)')
    plt.ylabel('Danceability')
    
    # Add a target box to highlight the "Summer Hit" zone
    plt.axvline(0.5, color='white', linestyle='--', alpha=0.5)
    plt.axhline(0.5, color='white', linestyle='--', alpha=0.5)
    
    save_path = output_dir / "03_hit_song_zone.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_audio_profile_radar(df: pd.DataFrame, output_dir: Path):
    """
    Creates an advanced Radar (Spider) Chart comparing the audio profile 
    of Smash Hits vs. Unpopular Tracks.
    """
    logging.info("Generating Audio Profile Radar Chart...")
    
    # Define the 0-1 scale features we want to map
    categories = ['danceability', 'energy', 'valence', 'acousticness', 'liveness']
    N = len(categories)
    
    # Calculate the average values for Hits (>80 popularity) and Flops (<20 popularity)
    hits_profile = df[df['popularity'] >= 80][categories].mean().values.flatten().tolist()
    flops_profile = df[df['popularity'] <= 20][categories].mean().values.flatten().tolist()
    
    # We need to repeat the first value at the end to close the circular graph
    hits_profile += hits_profile[:1]
    flops_profile += flops_profile[:1]
    
    # Calculate the angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories, size=12, fontweight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot Hits (Green)
    ax.plot(angles, hits_profile, linewidth=2, linestyle='solid', color='#1DB954', label='Smash Hits (>80 Pop)')
    ax.fill(angles, hits_profile, color='#1DB954', alpha=0.25)
    
    # Plot Flops (Red)
    ax.plot(angles, flops_profile, linewidth=2, linestyle='solid', color='#E22134', label='Unpopular (<20 Pop)')
    ax.fill(angles, flops_profile, color='#E22134', alpha=0.25)
    
    plt.title('Audio DNA Comparison: Hits vs. Flops', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    save_path = output_dir / "04_audio_dna_radar.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define file paths based on the production structure
    PROCESSED_DATA_PATH = Path("week_4/spotify_hit_analyzer/data/03_processed/spotify_cleaned.csv")
    FIGURES_DIR = Path("week_4/spotify_hit_analyzer/reports/figures/")

    try:
        # Ensure the output directory exists
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = load_data(PROCESSED_DATA_PATH)
        
        # Generate the visual report suite
        plot_correlation_heatmap(df, FIGURES_DIR)
        plot_energy_vs_acousticness(df, FIGURES_DIR)
        plot_vibe_vs_danceability(df, FIGURES_DIR)
        plot_audio_profile_radar(df, FIGURES_DIR)
        
        logging.info(f"Report generation complete! All visualizations successfully saved to {FIGURES_DIR.resolve()}")
        
    except FileNotFoundError:
        logging.error(f"Could not find the dataset at {PROCESSED_DATA_PATH}. Please run data_cleaner.py first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during visualization: {e}")