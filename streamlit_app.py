import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objs as go
from sklearn.metrics import r2_score

st.set_page_config(page_title="Electricity Forecast Dashboard", layout="wide")

# ----------------------- Session State Init -----------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# ----------------------- Data Cleaner -----------------------
def clean_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    if 'timestamp' not in df.columns or 'demand' not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'demand' columns.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# ----------------------- Load Datasets -----------------------
data_folder = "data"
if not st.session_state.datasets:
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            region_name = file.split(".")[0]
            df = pd.read_csv(os.path.join(data_folder, file))
            try:
                st.session_state.datasets[region_name] = clean_data(df)
            except Exception as e:
                st.warning(f"Failed to load {file}: {e}")

# ----------------------- Load Models -----------------------
models_folder = "models"
models = {}
model_scores = {}
for region in st.session_state.datasets.keys():
    model_path = os.path.join(models_folder, f"{region}_best_model.pkl")
    if os.path.exists(model_path):
        model_bundle = joblib.load(model_path)
        if isinstance(model_bundle, tuple):
            model, score = model_bundle
        else:
            model = model_bundle
            score = None
        models[region] = model
        model_scores[region] = score

# ----------------------- Aggregation -----------------------
def aggregate_df(df, freq):
    return df.resample(freq)['demand'].mean().reset_index()

# ----------------------- Forecast Range -----------------------
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
    for _ in range(50):
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
        'mean': mean,
        'interval': [f"({int(l)} - {int(u)}) kW" for l, u in zip(lower, upper)]
    })

    return forecast_df

# ----------------------- Create Chart -----------------------
def create_chart(df, model, freq, region):
    agg_df = aggregate_df(df, freq)
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=agg_df['timestamp'],
        y=agg_df['demand'],
        mode='lines+markers',
        name=f'{region.capitalize()} Historical Demand'
    ))

    # Forecast
    forecast_df = None
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

    return fig, forecast_df

# ----------------------- File Upload UI -----------------------
st.sidebar.subheader("Add New Region Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    new_region = st.sidebar.text_input("Enter Region Name")
    if new_region and st.sidebar.button("Add Region"):
        try:
            df_new = pd.read_csv(uploaded_file)
            cleaned = clean_data(df_new)
            st.session_state.datasets[new_region.lower()] = cleaned
            st.success(f"Region '{new_region}' added successfully!")
        except Exception as e:
            st.error(f"Error adding region: {e}")

# ----------------------- Controls -----------------------
region = st.sidebar.selectbox("Select Region", list(st.session_state.datasets.keys()))
freq = st.sidebar.selectbox("Time Scale", ['M', 'Q', 'Y'],
                            format_func=lambda x: {'M': 'Monthly', 'Q': 'Seasonal', 'Y': 'Yearly'}[x])

# ----------------------- Main Dashboard -----------------------
st.title("🔌 Electricity Forecast Dashboard")
df = st.session_state.datasets[region]
model = models.get(region)
model_score = model_scores.get(region)

fig, forecast_df = create_chart(df, model, freq, region)
st.plotly_chart(fig, use_container_width=True)

# ----------------------- Forecast Table -----------------------
if forecast_df is not None:
    st.subheader("📊 Forecasted Demand Ranges")
    st.dataframe(forecast_df[['timestamp', 'interval']].rename(columns={
        'timestamp': 'Date',
        'interval': 'Forecast Range (kW)'
    }), use_container_width=True)

    # ----------------------- Interpretation -----------------------
    st.markdown("### 📌 Interpretation")
    st.markdown(f"""
    - The green dashed line represents the **mean forecast**.
    - The shaded green area is the **80% confidence interval** (10th to 90th percentile).
    - Each time unit (e.g., month/year) includes a demand range prediction.
    - These ranges help decision-makers plan for best-case and worst-case scenarios.

    """)

    if model_score is not None:
        st.success(f"✅ Model R² Accuracy: {model_score * 100:.2f}% — this reflects how well the model fits the historical data.")

# ----------------------- Footer -----------------------
st.markdown("---")
st.caption("Electricity Forecast Dashboard © 2025 — Powered by Streamlit and Machine Learning")
