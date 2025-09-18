from flask import Blueprint, render_template, request
import pandas as pd
import joblib
import plotly
import plotly.graph_objs as go
import json
import os

main = Blueprint('main', __name__)

# -------------------------
# Load datasets
# -------------------------
files = ['kigoma.csv', 'katavi.csv', 'rukwa.csv']
datasets = {}
for file in files:
    df = pd.read_csv(f"data/{file}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    datasets[file.split('.')[0]] = df

# -------------------------
# Load models (optional)
# -------------------------
models = {}
for region in datasets.keys():
    model_path = f"app/models/{region}_best_model.pkl"
    if os.path.exists(model_path):
        models[region] = joblib.load(model_path)

# -------------------------
# Aggregate by frequency
# -------------------------
def aggregate_df(df, freq):
    return df.resample(freq)['demand'].mean().reset_index()

# -------------------------
# Create interactive Plotly chart
# -------------------------
def create_interactive_chart(region, freq):
    df = datasets[region]
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
    
    # ✅ Fixed plotly reference
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# -------------------------
# Routes
# -------------------------
@main.route('/')
def index():
    graphJSON = create_interactive_chart('kigoma', 'M')
    return render_template('index.html', graphJSON=graphJSON)

@main.route('/region/<region>')
def region_view(region):
    freq = request.args.get('freq', 'M')
    if region not in datasets:
        return f"Region {region} not found!"
    graphJSON = create_interactive_chart(region, freq)
    return render_template('region.html', region=region.capitalize(), graphJSON=graphJSON)
