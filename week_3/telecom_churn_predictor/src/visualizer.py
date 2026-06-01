import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set professional styling for corporate reports
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300 # High resolution for PDF reports and presentations

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the preprocessed dataset for visualization."""
    logging.info(f"Loading cleaned data from {filepath}")
    return pd.read_csv(filepath)

def plot_churn_distribution(df: pd.DataFrame, output_dir: Path):
    """Creates a donut chart showing the overall churn rate."""
    logging.info("Generating Churn Distribution chart...")
    
    plt.figure()
    counts = df['Churn'].value_counts()
    labels = ['Stayed (Active)', 'Churned (Left)']
    colors = ['#2ecc71', '#e74c3c']
    
    # Create a donut chart
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4))
    plt.title('Overall Customer Churn Rate', fontsize=16, fontweight='bold')
    
    save_path = output_dir / "01_churn_distribution.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_tenure_vs_churn(df: pd.DataFrame, output_dir: Path):
    """Creates a density plot showing when customers are most likely to leave."""
    logging.info("Generating Tenure vs. Churn chart...")
    
    plt.figure()
    sns.kdeplot(data=df[df['Churn'] == 0], x='tenure', label='Stayed', fill=True, color='#2ecc71', alpha=0.5)
    sns.kdeplot(data=df[df['Churn'] == 1], x='tenure', label='Churned', fill=True, color='#e74c3c', alpha=0.5)
    
    plt.title('Customer Churn Risk by Tenure (Months)', fontsize=16, fontweight='bold')
    plt.xlabel('Months with Company')
    plt.ylabel('Density of Customers')
    plt.legend()
    
    save_path = output_dir / "02_tenure_vs_churn.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_contract_type_impact(df: pd.DataFrame, output_dir: Path):
    """Creates a bar chart analyzing the impact of contract types on churn."""
    logging.info("Generating Contract Type Impact chart...")
    
    plt.figure()
    # Calculate churn rate percentage per contract type
    contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False) * 100
    
    sns.barplot(x=contract_churn.index, y=contract_churn.values, palette='Reds_r')
    
    plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold')
    plt.xlabel('Contract Type')
    plt.ylabel('Churn Rate (%)')
    
    # Add percentage labels on top of the bars
    for i, v in enumerate(contract_churn.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=12)
    
    save_path = output_dir / "03_contract_impact.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_monthly_charges_impact(df: pd.DataFrame, output_dir: Path):
    """Creates a boxplot to show if higher bills push customers to cancel."""
    logging.info("Generating Monthly Charges Impact chart...")
    
    plt.figure()
    # Map 1/0 back to text for cleaner chart labels
    plot_df = df.copy()
    plot_df['Status'] = plot_df['Churn'].map({0: 'Stayed', 1: 'Churned'})
    
    sns.boxplot(x='Status', y='MonthlyCharges', data=plot_df, palette=['#2ecc71', '#e74c3c'])
    
    plt.title('Impact of Monthly Charges on Churn', fontsize=16, fontweight='bold')
    plt.xlabel('Customer Status')
    plt.ylabel('Monthly Bill ($)')
    
    save_path = output_dir / "04_monthly_charges_impact.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Creates a correlation heatmap of all numerical features against Churn."""
    logging.info("Generating Correlation Heatmap...")
    
    plt.figure(figsize=(8, 6))
    
    # Select only numerical columns for correlation
    num_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    correlation_matrix = num_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix: Numerical Features vs. Churn', fontsize=16, fontweight='bold')
    
    save_path = output_dir / "05_correlation_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define file paths based on the production structure
    PROCESSED_DATA_PATH = Path("week_3/telecom_churn_predictor/data/03_processed/telco_churn_cleaned.csv")
    FIGURES_DIR = Path("week_3/telecom_churn_predictor/reports/figures/")

    try:
        # Ensure the figures directory exists
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load the cleaned data
        df = load_data(PROCESSED_DATA_PATH)
        
        # Generate the visual report suite
        plot_churn_distribution(df, FIGURES_DIR)
        plot_tenure_vs_churn(df, FIGURES_DIR)
        plot_contract_type_impact(df, FIGURES_DIR)
        plot_monthly_charges_impact(df, FIGURES_DIR)
        plot_correlation_heatmap(df, FIGURES_DIR)
        
        logging.info(f"Report generation complete! All visualizations successfully saved to {FIGURES_DIR.resolve()}")
        
    except FileNotFoundError:
        logging.error(f"Could not find the dataset at {PROCESSED_DATA_PATH}. Please run data_cleaner.py first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during visualization: {e}")