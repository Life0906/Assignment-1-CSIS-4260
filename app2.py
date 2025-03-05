import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import time
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load dataset
@st.cache_data
def load_data():
    return pd.read_parquet("data.parquet")

df = load_data()

# Benchmarking function
def benchmark_format(df, format_name, compression=None):
    file_name = f"benchmark_data.{format_name}"
    start_time = time.time()
    if format_name == "csv":
        df.to_csv(file_name, index=False)
    elif format_name == "parquet":
        df.to_parquet(file_name, index=False, compression=compression)
    write_time = time.time() - start_time
    file_size = os.path.getsize(file_name) / (1024 * 1024)
    start_time = time.time()
    if format_name == "csv":
        _ = pd.read_csv(file_name)
    elif format_name == "parquet":
        _ = pd.read_parquet(file_name)
    read_time = time.time() - start_time
    return write_time, read_time, file_size

# Benchmark Results
formats = [("csv", None), ("parquet", None), ("parquet", "snappy"), ("parquet", "gzip"), ("parquet", "brotli")]
def run_benchmarks(df_scaled):
    results = {"Format": [], "Compression": [], "Write Time (s)": [], "Read Time (s)": [], "File Size (MB)": []}
    for fmt, comp in formats:
        write_time, read_time, file_size = benchmark_format(df_scaled, fmt, comp)
        results["Format"].append(fmt)
        results["Compression"].append(comp if comp else "None")
        results["Write Time (s)"].append(write_time)
        results["Read Time (s)"].append(read_time)
        results["File Size (MB)"].append(file_size)
    return pd.DataFrame(results)

scales = {"1x": df, "10x": pd.concat([df] * 10, ignore_index=True), "100x": pd.concat([df] * 100, ignore_index=True)}

# Sidebar
st.sidebar.title("Dashboard Options")
section = st.sidebar.radio("Choose Section", ["Storage Benchmarking", "Stock Market & Predictions"])

if section == "Storage Benchmarking":
    st.title("Data Storage Benchmarking")
    scale_choice = st.selectbox("Select Scale", list(scales.keys()))
    st.dataframe(run_benchmarks(scales[scale_choice]))

else:
    st.title("Stock Market Dashboard")
    stock_options = df['Name'].unique()
    selected_stock = st.selectbox("Choose a stock", stock_options)
    chart_type = st.radio("Chart Type", ["Line Chart", "Candlestick Chart"])
    stock_data = df[df['Name'] == selected_stock]

    fig = go.Figure()
    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], mode='lines', name='Closing Price'))
    else:
        fig.add_trace(go.Candlestick(x=stock_data['date'], open=stock_data['open'], high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], name='Candlestick'))
    
    st.plotly_chart(fig)
    
    # Placeholder for predictions (XGBoost & Random Forest)
    st.subheader("Predictions (XGBoost & Random Forest)")
    st.write("Coming Soon...")
