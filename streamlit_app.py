import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import os
import numpy as np

st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide")

# -----------------------
# Initialize session state
# -----------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# -----------------------
# Load datasets
# -----------------------
data_folder = "data"
if not st.session_state.datasets:
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            region_name = file.split(".")[0]
            df = pd.read_csv(os.path.join(data_folder, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            st.session_state.datasets[region_name] = df

# -----------------------
# Load models
# -----------------------
models_folder = "models"
models = {}
for region in st.session_state.datasets.keys():
    model_path = os.path.join(models_folder, f"{region}_best_model.pkl")
    if os.path.exists(model_path):
        models[region] = joblib.load(model_path)

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

    X_pred = pd.DataFrame({
        'month': future_dates.month,
        'quarter': future_dates.quarter,
        'year': future_dates.year
    })

    preds = []
    for _ in range(50):  # simulate 50 samples
        noise = np.random.normal(0, 0.5, len(X_pred))
        pred = model.predict(X_pred) + noise
        preds.append(pred)

    preds = np.array(preds)
    lower = np.percentile(preds, 10, axis=0)
    upper = np.percentile(preds, 90, axis=0)
    mean = preds.mean(axis=0)

    forecast_df = pd.DataFrame({
        'timestamp': future_dates,
        'lower': lower,
        'upper': upper,
        'mean': mean
    })

    return forecast_df

# -----------------------
# Chart
# -----------------------
def create_chart(df, model, freq, region):
    agg_df = aggregate_df(df, freq)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg_df['timestamp'],
        y=agg_df['demand'],
        mode='lines+markers',
        name=f'{region.capitalize()} Historical Demand'
    ))

    if model is not None:
        forecast_df = forecast_range(model, df, freq)
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['mean'],
            mode='lines+markers',
            name="Forecast (mean)",
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['upper'],
            mode='lines',
            name='Forecast Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
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

    fig.update_layout(
        title=f"{region.capitalize()} Electricity Demand Forecast - {freq} View",
        xaxis_title='Time',
        yaxis_title='Demand (kW)',
        hovermode='x unified'
    )
    return fig

# -----------------------
# Upload New Region
# -----------------------
st.sidebar.subheader("Add New Region Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    new_region = st.sidebar.text_input("Enter Region Name")
    if new_region and st.sidebar.button("Add Region"):
        df_new = pd.read_csv(uploaded_file)
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
        df_new.set_index('timestamp', inplace=True)
        st.session_state.datasets[new_region.lower()] = df_new
        st.success(f"Region '{new_region}' added successfully!")

# -----------------------
# Sidebar Controls
# -----------------------
region = st.sidebar.selectbox("Select Region", list(st.session_state.datasets.keys()))
freq = st.sidebar.selectbox("Time Scale", ['M', 'Q', 'Y'],
                            format_func=lambda x: {'M': 'Monthly', 'Q': 'Seasonal', 'Y': 'Yearly'}[x])

# -----------------------
# Main Chart & Forecast Text Output
# -----------------------
st.title("🔌 Electricity Forecast Dashboard")
df = st.session_state.datasets[region]
model = models.get(region)

fig = create_chart(df, model, freq, region)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Forecast Table with Intervals
# -----------------------
if model:
    forecast_df = forecast_range(model, df, freq)

    st.subheader("📊 Forecasted Demand Ranges")
    
    def format_label(ts, freq):
        if freq == 'M':
            return ts.strftime('%b %Y')
        elif freq == 'Q':
            return f"Q{((ts.month-1)//3)+1} {ts.year}"
        elif freq == 'Y':
            return str(ts.year)

    forecast_df['label'] = forecast_df['timestamp'].apply(lambda ts: format_label(ts, freq))
    forecast_df['Range (kW)'] = forecast_df.apply(lambda row: f"{row['lower']:.0f} – {row['upper']:.0f} kW", axis=1)

    st.table(forecast_df[['label', 'Range (kW)']].rename(columns={'label': 'Period'}))
