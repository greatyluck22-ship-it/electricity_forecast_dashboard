import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import os

st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide")

# -----------------------
# Load datasets dynamically
# -----------------------
data_folder = "data"
datasets = {}
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        region_name = file.split(".")[0]
        df = pd.read_csv(os.path.join(data_folder, file))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        datasets[region_name] = df

# -----------------------
# Load models dynamically
# -----------------------
models_folder = "models"
models = {}
for region in datasets.keys():
    model_path = os.path.join(models_folder, f"{region}_best_model.pkl")
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
freq = st.sidebar.selectbox(
    "Time Scale", 
    ['M','W','Q','Y'],
    format_func=lambda x: {'M':'Monthly','W':'Weekly','Q':'Seasonal','Y':'Yearly'}[x]
)

# -----------------------
# Upload new CSV for additional regions
# -----------------------
st.sidebar.subheader("Add New Region Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    new_region = st.sidebar.text_input("Enter Region Name")
    if new_region and st.sidebar.button("Add Region"):
        df_new = pd.read_csv(uploaded_file)
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
        df_new.set_index('timestamp', inplace=True)
        datasets[new_region.lower()] = df_new
        st.success(f"Region '{new_region}' added successfully!")

# -----------------------
# Display Chart
# -----------------------
st.title("Electricity Forecast Dashboard")
st.plotly_chart(create_chart(datasets[region], freq, region), use_container_width=True)
