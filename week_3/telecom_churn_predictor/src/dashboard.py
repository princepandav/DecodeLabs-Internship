import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. Configure the Web Page (Must be the first Streamlit command)
st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load the Data with Caching for High Performance
@st.cache_data
def load_data():
    filepath = Path("../data/03_processed/telco_churn_cleaned.csv")
    return pd.read_csv(filepath)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found. Please run data_cleaner.py first.")
    st.stop()

# 3. Sidebar: Dynamic User Controls
st.sidebar.header("🔍 Dynamic Filters")
st.sidebar.markdown("Use these filters to instantly update the dashboard.")

# Filter: Contract Type
contract_options = df['Contract'].unique().tolist()
selected_contracts = st.sidebar.multiselect(
    "Filter by Contract Type:",
    options=contract_options,
    default=contract_options
)

# Filter: Internet Service
internet_options = df['InternetService'].unique().tolist()
selected_internet = st.sidebar.multiselect(
    "Filter by Internet Service:",
    options=internet_options,
    default=internet_options
)

# Filter: Tenure Slider
min_tenure = int(df['tenure'].min())
max_tenure = int(df['tenure'].max())
selected_tenure = st.sidebar.slider(
    "Select Tenure Range (Months):",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure)
)

# Apply the filters to the dataframe dynamically
filtered_df = df[
    (df['Contract'].isin(selected_contracts)) &
    (df['InternetService'].isin(selected_internet)) &
    (df['tenure'].between(selected_tenure[0], selected_tenure[1]))
]

# 4. Main Dashboard Area
st.title("📡 Telecom Customer Churn Analytics")
st.markdown("Analyze customer retention metrics and identify high-risk subscriber segments in real-time.")

# High-Level KPIs
col1, col2, col3, col4 = st.columns(4)
total_customers = len(filtered_df)

if total_customers > 0:
    churn_rate = (filtered_df['Churn'].mean() * 100)
    avg_tenure = filtered_df['tenure'].mean()
    avg_monthly = filtered_df['MonthlyCharges'].mean()
else:
    churn_rate = avg_tenure = avg_monthly = 0

col1.metric("Total Customers (Filtered)", f"{total_customers:,}")
col2.metric("Active Churn Rate", f"{churn_rate:.1f}%")
col3.metric("Avg. Tenure", f"{avg_tenure:.1f} Months")
col4.metric("Avg. Monthly Bill", f"${avg_monthly:.2f}")

st.divider()

if total_customers == 0:
    st.warning("No data matches your current filter selection. Please adjust the sidebar filters.")
    st.stop()

# 5. Interactive Visualizations
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Overall Churn Status")
    # Convert binary to text for the pie chart
    pie_data = filtered_df['Churn'].value_counts().reset_index()
    pie_data['Churn'] = pie_data['Churn'].map({0: 'Stayed', 1: 'Churned'})
    
    fig_pie = px.pie(
        pie_data, 
        names='Churn', 
        values='count',
        hole=0.4,
        color='Churn',
        color_discrete_map={'Stayed': '#2ecc71', 'Churned': '#e74c3c'}
    )
    st.plotly_chart(fig_pie, width="stretch") # Using 'width' to avoid deprecation warnings

with row1_col2:
    st.subheader("Churn Rate by Contract Type")
    # Calculate churn rate per contract type
    contract_churn = filtered_df.groupby('Contract')['Churn'].mean().reset_index()
    contract_churn['Churn Rate (%)'] = contract_churn['Churn'] * 100
    
    fig_bar = px.bar(
        contract_churn, 
        x='Contract', 
        y='Churn Rate (%)',
        text_auto='.1f',
        color='Churn Rate (%)',
        color_continuous_scale='Reds'
    )
    fig_bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_bar, width="stretch")

st.divider()

st.subheader("Financial Impact: Monthly Charges vs. Churn")
# Create a copy for the boxplot to change labels without affecting the main dataframe
box_df = filtered_df.copy()
box_df['Status'] = box_df['Churn'].map({0: 'Stayed', 1: 'Churned'})

fig_box = px.box(
    box_df,
    x="Status",
    y="MonthlyCharges",
    color="Status",
    color_discrete_map={'Stayed': '#2ecc71', 'Churned': '#e74c3c'},
    points="all" # Shows individual data points alongside the box
)
st.plotly_chart(fig_box, width="stretch")