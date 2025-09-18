# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import os

st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide")

# -----------------------
# Load datasets
# -----------------------
files = ['kigoma.csv','katavi.csv','rukwa.csv']
datasets = {}
for file in files:
    df = pd.read_csv(f"app/data/{file}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    datasets[file.split('.')[0]] = df

# -----------------------
# Load models (optional)
# -----------------------
models = {}
for region in datasets.keys():
    model_path = f"app/models/{region}_best_model.pkl"
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
region = st.sidebar.selectbox("Select Region", list(datasets.keys()))
freq = st.sidebar.selectbox("Time Scale", ['M','W','Q','Y'], format_func=lambda x: {'M':'Monthly','W':'Weekly','Q':'Seasonal','Y':'Yearly'}[x])

# -----------------------
# Display Chart
# -----------------------
st.title("Electricity Forecast Dashboard")
st.plotly_chart(create_chart(datasets[region], freq, region), use_container_width=True)
