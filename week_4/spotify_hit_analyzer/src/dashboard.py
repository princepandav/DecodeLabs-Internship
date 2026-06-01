import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 1. Configure the Web Page (Must be the first Streamlit command)
st.set_page_config(
    page_title="Spotify Hit Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load the Data with Caching for High Performance
@st.cache_data
def load_data():
    filepath = Path("../data/03_processed/spotify_cleaned.csv")
    return pd.read_csv(filepath)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found. Please run data_cleaner.py first.")
    st.stop()

# 3. Sidebar: Dynamic User Controls
st.sidebar.header("🎛️ Audio Mixing Board")
st.sidebar.markdown("Filter the global Spotify database to uncover niche trends.")

# Filter: Genre
# Limiting to top 50 genres for UI performance if the dataset has hundreds
top_genres = df['track_genre'].value_counts().head(50).index.tolist()
selected_genres = st.sidebar.multiselect(
    "Filter by Genre (Top 50):",
    options=top_genres,
    default=top_genres[:5] # Default to the top 5 to avoid overwhelming the initial load
)

# Filter: Popularity Slider
min_pop = int(df['popularity'].min())
max_pop = int(df['popularity'].max())
selected_pop = st.sidebar.slider(
    "Select Popularity Range:",
    min_value=min_pop,
    max_value=max_pop,
    value=(50, max_pop) # Default to looking at somewhat popular tracks
)

# Filter: Explicit Tracks
explicit_mapping = {0: "Clean", 1: "Explicit"}
df['explicit_label'] = df['explicit'].map(explicit_mapping)
explicit_options = df['explicit_label'].dropna().unique().tolist()
selected_explicit = st.sidebar.multiselect(
    "Content Type:",
    options=explicit_options,
    default=explicit_options
)

# Apply filters dynamically
filtered_df = df[
    (df['track_genre'].isin(selected_genres)) &
    (df['popularity'].between(selected_pop[0], selected_pop[1])) &
    (df['explicit_label'].isin(selected_explicit))
]

# 4. Main Dashboard Area
st.title("🎵 Spotify Audio DNA Dashboard")
st.markdown("Deconstruct the mathematical formula behind thousands of tracks.")

# High-Level KPIs
col1, col2, col3, col4 = st.columns(4)
total_tracks = len(filtered_df)

if total_tracks > 0:
    avg_pop = filtered_df['popularity'].mean()
    avg_energy = filtered_df['energy'].mean()
    avg_dance = filtered_df['danceability'].mean()
else:
    avg_pop = avg_energy = avg_dance = 0

col1.metric("Tracks in Selection", f"{total_tracks:,}")
col2.metric("Avg. Popularity", f"{avg_pop:.1f} / 100")
col3.metric("Avg. Energy", f"{avg_energy:.2f}")
col4.metric("Avg. Danceability", f"{avg_dance:.2f}")

st.divider()

if total_tracks == 0:
    st.warning("No tracks match your current mix. Adjust the sidebar filters.")
    st.stop()

# 5. Interactive Visualizations
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("The Vibe Matrix: Happiness vs. Danceability")
    # Using a scatter plot with a trendline for interactive exploration
    # Sampling down to 2000 points if the dataset is massive so the browser doesn't crash
    plot_df = filtered_df.sample(min(2000, len(filtered_df)), random_state=42) 
    
    fig_scatter = px.scatter(
        plot_df, 
        x="valence", 
        y="danceability", 
        color="popularity",
        size="popularity",
        hover_data=['track_name', 'artists'],
        color_continuous_scale="magma",
        labels={'valence': 'Valence (Happiness)', 'danceability': 'Danceability'}
    )
    st.plotly_chart(fig_scatter, width="stretch")

with row1_col2:
    st.subheader("Audio DNA Profile (Selected vs. Global Avg)")
    
    categories = ['danceability', 'energy', 'valence', 'acousticness', 'liveness']
    
    # Calculate means
    filtered_means = [filtered_df[cat].mean() for cat in categories]
    global_means = [df[cat].mean() for cat in categories]
    
    # Close the radar chart loop
    filtered_means.append(filtered_means[0])
    global_means.append(global_means[0])
    categories_loop = categories + [categories[0]]
    
    fig_radar = go.Figure()
    
    # Global Baseline Trace
    fig_radar.add_trace(go.Scatterpolar(
        r=global_means,
        theta=categories_loop,
        fill='toself',
        name='Global Average',
        line_color='gray',
        opacity=0.5
    ))
    
    # Filtered Selection Trace
    fig_radar.add_trace(go.Scatterpolar(
        r=filtered_means,
        theta=categories_loop,
        fill='toself',
        name='Your Filtered Mix',
        line_color='#1DB954' # Spotify Green
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    st.plotly_chart(fig_radar, width="stretch")

st.divider()

# Full-width chart for Genre comparison
st.subheader("Popularity Distribution Across Selected Genres")
fig_box = px.box(
    filtered_df,
    x="track_genre",
    y="popularity",
    color="track_genre",
    points="outliers",
    labels={'track_genre': 'Genre', 'popularity': 'Popularity Score'}
)
fig_box.update_layout(showlegend=False)
st.plotly_chart(fig_box, width="stretch")