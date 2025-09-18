import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import os
from glob import glob

st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide")

# -----------------------
# Load datasets dynamically
# -----------------------
datasets = {}
csv_files = glob("data/*.csv")  # ✅ load all CSVs from /data
for file_path in csv_files:
    region = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "kigoma"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['region'] = region  # ✅ add region column
    datasets[region] = df

# ✅ Combine all into one global dataset
global_df = pd.concat(datasets.values(), axis=0)

# -----------------------
# Load models (optional)
# -----------------------
models = {}
for region in datasets.keys():
    model_path = os.path.join("models", f"{region}_best_model.pkl")
    if os.path.exists(model_path):
        models[region] = joblib.load(model_path)

# -----------------------
# Aggregate by frequency
# -----------------------
def aggregate_df(df, freq):
    return df.resample(freq)['demand'].mean().reset_index()

# -----------------------
# Create Plotly chart
# -----------------------
def create_chart(df, freq, region):
    agg_df = aggregate_df(df, freq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg_df['timestamp'],
        y=agg_df['demand'],
        mode='lines+markers',
        name=f'{region.capitalize()} Demand'
    ))
    fig.update_layout(
        title=f'{region.capitalize()} Electricity Demand - {freq} View',
        xaxis_title='Time',
        yaxis_title='Demand (kW)',
        hovermode='x unified'
    )
    return fig

# -----------------------
# Streamlit Sidebar
# -----------------------
st.sidebar.title("Settings")

# ✅ Add "All Regions (Global)" option
region_options = list(datasets.keys()) + ["All Regions"]
region = st.sidebar.selectbox("Select Region", region_options)

freq = st.sidebar.selectbox(
    "Time Scale",
    ['M','W','Q','Y'],
    format_func=lambda x: {'M':'Monthly','W':'Weekly','Q':'Seasonal','Y':'Yearly'}[x]
)

# -----------------------
# Display Chart
# -----------------------
st.title("Electricity Forecast Dashboard")

if region == "All Regions":
    # Plot global aggregated demand
    agg_df = global_df.groupby('timestamp').sum().resample(freq)['demand'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg_df['timestamp'],
        y=agg_df['demand'],
        mode='lines+markers',
        name="Global Demand"
    ))
    fig.update_layout(
        title=f"All Regions Electricity Demand - {freq} View",
        xaxis_title="Time",
        yaxis_title="Demand (kW)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(create_chart(datasets[region], freq, region), use_container_width=True)
