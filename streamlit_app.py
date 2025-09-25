import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import os
import numpy as np
from sklearn.metrics import r2_score

st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide")

# -----------------------
# Initialize session state
# -----------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# -----------------------
# Data cleaning function
# -----------------------
def clean_data(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp', 'demand'], inplace=True)
    df = df[df['demand'] >= 0]  # remove negative demand if any
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    return df

# -----------------------
# Load datasets and clean
# -----------------------
data_folder = "data"
if not st.session_state.datasets:
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            region_name = file.split(".")[0]
            df = pd.read_csv(os.path.join(data_folder, file))
            df = clean_data(df)
            st.session_state.datasets[region_name] = df

# -----------------------
# Load models & accuracy if saved as tuple (model, score)
# -----------------------
models_folder = "models"
models = {}
model_scores = {}
for region in st.session_state.datasets.keys():
    model_path = os.path.join(models_folder, f"{region}_best_model.pkl")
    if os.path.exists(model_path):
        loaded = joblib.load(model_path)
        if isinstance(loaded, tuple):
            model, score = loaded
            models[region] = model
            model_scores[region] = score
        else:
            models[region] = loaded
            model_scores[region] = None

# -----------------------
# Aggregation
# -----------------------
def aggregate_df(df, freq):
    return df.resample(freq)['demand'].mean().reset_index()

# -----------------------
# Forecast future demand range
# -----------------------
def forecast_range(model, df, freq, periods=12):
    last_date = df.index.max()

    if freq == 'M':
        future_dates = pd.date_range(last_date, periods=periods + 1, freq='MS')[1:]
    elif freq == 'Q':
        future_dates = pd.date_range(last_date, periods=periods + 1, freq='QS')[1:]
    elif freq == 'Y':
        future_dates = pd.date_range(last_date, periods=periods + 1, freq='YS')[1:]
    else:
        future_dates = pd.date_range(last_date, periods=periods + 1, freq='MS')[1:]  # default monthly

    X_pred = pd.DataFrame({
        'month': future_dates.month,
        'quarter': future_dates.quarter,
        'year': future_dates.year
    })

    # Predict multiple times with noise to simulate uncertainty
    preds = []
    for _ in range(50):
        noise = np.random.normal(0, 0.5, len(X_pred))
        pred = model.predict(X_pred) + noise
        preds.append(pred)

    preds = np.array(preds)
    lower = np.percentile(preds, 10, axis=0)
    upper = np.percentile(preds, 90, axis=0)

    forecast_df = pd.DataFrame({
        'timestamp': future_dates,
        'lower': lower,
        'upper': upper,
        'mean': preds.mean(axis=0)
    })

    return forecast_df

# -----------------------
# Create Plotly chart with intervals
# -----------------------
def create_chart(df, model, freq, region):
    agg_df = aggregate_df(df, freq)

    fig = go.Figure()

    # Historical demand
    fig.add_trace(go.Scatter(
        x=agg_df['timestamp'],
        y=agg_df['demand'],
        mode='lines+markers',
        name=f'{region.capitalize()} Historical Demand'
    ))

    if model is not None:
        forecast_df = forecast_range(model, df, freq)

        # Forecast mean
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['mean'],
            mode='lines+markers',
            name='Forecast (mean)',
            line=dict(color='green', dash='dash')
        ))

        # Upper bound
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['upper'],
            mode='lines',
            name='Forecast Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))

        # Lower bound with fill to upper bound for interval shading
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['lower'],
            mode='lines',
            name='Forecast Lower Bound',
            fill='tonexty',
            fillcolor='rgba(0,200,0,0.2)',
            line=dict(width=0),
            showlegend=True
        ))

        return fig, forecast_df
    else:
        return fig, None

# -----------------------
# Format intervals for display below chart
# -----------------------
def format_intervals(forecast_df, freq):
    lines = []
    for _, row in forecast_df.iterrows():
        ts = row['timestamp']
        if freq == 'Y':
            label = ts.year
        elif freq == 'Q':
            label = f"Q{((ts.month - 1) // 3) + 1} {ts.year}"
        else:  # Monthly
            label = ts.strftime("%b %Y")

        lines.append(f"{label}: ({row['lower']:.1f} - {row['upper']:.1f}) kW")

    return lines

# -----------------------
# Upload New Region with cleaning
# -----------------------
st.sidebar.subheader("Add New Region Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    new_region = st.sidebar.text_input("Enter Region Name")
    if new_region and st.sidebar.button("Add Region"):
        df_new = pd.read_csv(uploaded_file)
        df_new = clean_data(df_new)
        st.session_state.datasets[new_region.lower()] = df_new
        st.success(f"Region '{new_region}' added successfully!")

# -----------------------
# Sidebar Controls
# -----------------------
region = st.sidebar.selectbox("Select Region", list(st.session_state.datasets.keys()))
freq = st.sidebar.selectbox("Time Scale", ['M', 'Q', 'Y'],
                            format_func=lambda x: {'M': 'Monthly', 'Q': 'Seasonal', 'Y': 'Yearly'}[x])

# -----------------------
# Main App Content
# -----------------------
st.title("🔌 Electricity Forecast Dashboard")

df = st.session_state.datasets[region]
model = models.get(region)
score = model_scores.get(region)

fig, forecast_df = create_chart(df, model, freq, region)

st.plotly_chart(fig, use_container_width=True)

# Show forecast intervals below chart
if forecast_df is not None:
    st.subheader("Forecast Demand Intervals")
    intervals = format_intervals(forecast_df, freq)
    for line in intervals:
        st.write(line)

# Show model accuracy
if score is not None:
    st.markdown(f"**Model R² Accuracy:** {score*100:.2f}%")
else:
    st.markdown("**Model Accuracy:** Not Available")

# Add interpretation/help text
st.markdown("""
---
### Interpretation

- The green dashed line shows the mean forecasted electricity demand.
- The shaded green area shows the uncertainty interval (10th to 90th percentile), reflecting prediction variability.
- Historical data is shown with blue lines and markers.
- Model accuracy (R²) indicates how well the model fits past data. Values closer to 100% indicate better fit.
- Data is cleaned on upload by removing missing timestamps and negative demands.

""")
