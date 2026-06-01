import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. Configure the Web Page
st.set_page_config(
    page_title="No-Show Optimizer",
    page_icon="🏥",
    layout="wide"
)

# 2. Load the Data
@st.cache_data
def load_data():
    filepath = Path("week_2/medical_no_show_optimizer/data/03_processed/medical_no_shows_cleaned.csv")
    return pd.read_csv(filepath)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found. Please ensure you have run data_cleaner.py first.")
    st.stop()

# 3. Build the Dashboard Header & KPIs
st.title("🏥 Medical Appointment No-Show Dashboard")
st.markdown("Analyze patient attendance patterns to optimize clinic scheduling and reduce revenue loss.")

# Calculate High-Level Metrics
total_appointments = len(df)
no_show_rate = (df['No_Show'].mean() * 100)
avg_wait_time = df['AwaitingTime'].mean()

# Display Metrics in columns
col1, col2, col3 = st.columns(3)
col1.metric("Total Appointments", f"{total_appointments:,}")
col2.metric("Overall No-Show Rate", f"{no_show_rate:.1f}%")
col3.metric("Average Wait Time", f"{avg_wait_time:.1f} Days")

st.divider()

# 4. Build Interactive Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Attendance Overview")
    # Interactive Pie Chart
    pie_data = df['No_Show'].value_counts().reset_index()
    pie_data['No_Show'] = pie_data['No_Show'].map({0: 'Showed Up', 1: 'No-Show'})
    fig_pie = px.pie(
        pie_data, 
        names='No_Show', 
        values='count', 
        color='No_Show',
        color_discrete_map={'Showed Up': '#2ecc71', 'No-Show': '#e74c3c'},
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Impact of SMS Reminders")
    # Interactive Bar Chart
    sms_impact = df.groupby('SMS_received')['No_Show'].mean().reset_index()
    sms_impact['No_Show Rate (%)'] = sms_impact['No_Show'] * 100
    sms_impact['SMS_received'] = sms_impact['SMS_received'].map({0: 'No SMS', 1: 'Received SMS'})
    
    fig_sms = px.bar(
        sms_impact, 
        x='SMS_received', 
        y='No_Show Rate (%)',
        color='SMS_received',
        color_discrete_map={'No SMS': '#3498db', 'Received SMS': '#f1c40f'},
        text_auto='.1f'
    )
    st.plotly_chart(fig_sms, use_container_width=True)

st.divider()

# Full-width charts
st.subheader("Age Distribution: Shows vs. No-Shows")
# Map binary to text for cleaner graph labels
df['Status'] = df['No_Show'].map({0: 'Showed Up', 1: 'No-Show'})
fig_age = px.histogram(
    df, 
    x="Age", 
    color="Status", 
    barmode="overlay",
    color_discrete_map={'Showed Up': '#2ecc71', 'No-Show': '#e74c3c'},
    nbins=80,
    opacity=0.7
)
st.plotly_chart(fig_age, use_container_width=True)

st.subheader("Waiting Time Impact on Attendance")
# Filter extreme outliers just for the visual
filtered_df = df[df['AwaitingTime'] <= 100]
fig_wait = px.box(
    filtered_df, 
    x="Status", 
    y="AwaitingTime", 
    color="Status",
    color_discrete_map={'Showed Up': '#2ecc71', 'No-Show': '#e74c3c'}
)
st.plotly_chart(fig_wait, use_container_width=True)