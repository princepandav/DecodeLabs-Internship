import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set professional styling for all plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300 

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the preprocessed CSV dataset."""
    logging.info(f"Loading cleaned data from {filepath}")
    return pd.read_csv(filepath)

def plot_overall_no_show_rate(df: pd.DataFrame, output_dir: Path):
    """Creates a pie chart showing the overall no-show rate."""
    logging.info("Generating overall no-show rate chart...")
    
    plt.figure()
    counts = df['No_Show'].value_counts()
    labels = ['Showed Up', 'No-Show']
    colors = ['#2ecc71', '#e74c3c']
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=[0, 0.1])
    plt.title('Overall Patient No-Show Rate', fontsize=16, fontweight='bold')
    
    save_path = output_dir / "01_overall_rate.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_no_show_by_age(df: pd.DataFrame, output_dir: Path):
    """Creates a density plot comparing the age distribution of shows vs. no-shows."""
    logging.info("Generating age distribution chart...")
    
    plt.figure()
    sns.kdeplot(data=df[df['No_Show'] == 0], x='Age', label='Showed Up', fill=True, color='#2ecc71', alpha=0.5)
    sns.kdeplot(data=df[df['No_Show'] == 1], x='Age', label='No-Show', fill=True, color='#e74c3c', alpha=0.5)
    
    plt.title('Age Distribution: Shows vs. No-Shows', fontsize=16, fontweight='bold')
    plt.xlabel('Patient Age')
    plt.ylabel('Density')
    plt.legend()
    
    save_path = output_dir / "02_age_distribution.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_awaiting_time_impact(df: pd.DataFrame, output_dir: Path):
    """Creates a boxplot to show how waiting time impacts the likelihood of a no-show."""
    logging.info("Generating awaiting time impact chart...")
    
    plt.figure()
    filtered_df = df[df['AwaitingTime'] <= 100].copy()
    filtered_df['Status'] = filtered_df['No_Show'].map({0: 'Showed Up', 1: 'No-Show'})
    
    sns.boxplot(x='Status', y='AwaitingTime', data=filtered_df, palette=['#2ecc71', '#e74c3c'])
    
    plt.title('Impact of Waiting Time on Appointment Attendance', fontsize=16, fontweight='bold')
    plt.xlabel('Appointment Status')
    plt.ylabel('Days Between Scheduling and Appointment')
    
    save_path = output_dir / "03_awaiting_time_impact.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_sms_reminder_effectiveness(df: pd.DataFrame, output_dir: Path):
    """Creates a bar chart analyzing if SMS reminders actually work."""
    logging.info("Generating SMS reminder effectiveness chart...")
    
    plt.figure()
    sms_impact = df.groupby('SMS_received')['No_Show'].mean() * 100
    
    sns.barplot(x=sms_impact.index, y=sms_impact.values, palette=['#3498db', '#f1c40f'])
    
    plt.title('No-Show Rate based on SMS Reminders', fontsize=16, fontweight='bold')
    plt.xlabel('SMS Received (0 = No, 1 = Yes)')
    plt.ylabel('No-Show Percentage (%)')
    plt.xticks(ticks=[0, 1], labels=['Did Not Receive SMS', 'Received SMS'])
    
    for i, v in enumerate(sms_impact.values):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
    
    save_path = output_dir / "04_sms_effectiveness.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    PROCESSED_DATA_PATH = Path("week_2/medical_no_show_optimizer/data/03_processed/medical_no_shows_cleaned.csv")
    FIGURES_DIR = Path("week_2/medical_no_show_optimizer/reports/figures/")

    try:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        df = load_data(PROCESSED_DATA_PATH)

        plot_overall_no_show_rate(df, FIGURES_DIR)
        plot_no_show_by_age(df, FIGURES_DIR)
        plot_awaiting_time_impact(df, FIGURES_DIR)
        plot_sms_reminder_effectiveness(df, FIGURES_DIR)
        
        logging.info(f"All visualizations successfully saved to {FIGURES_DIR.resolve()}")
        
    except FileNotFoundError:
        logging.error(f"Could not find the dataset at {PROCESSED_DATA_PATH}. Run data_cleaner.py first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during visualization: {e}")