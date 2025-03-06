import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import joblib

# Load precomputed benchmarking results
benchmark_1x = pd.read_csv("benchmark_results.csv")
benchmark_10x = pd.read_csv("benchmark_results_10x.csv")
benchmark_100x = pd.read_csv("benchmark_results_100x.csv")

# Load stock data
stock_data = pd.read_parquet("data.parquet")

# Streamlit App
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("Stock Market Dashboard")

# Sidebar for Stock Selection
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.selectbox("Choose a stock:", stock_data['name'].unique())

# Filter Data for Selected Stock
stock_df = stock_data[stock_data['name'] == selected_stock]

# Section A: Storage Benchmarking
st.subheader("📊 Storage Format Benchmarking")
tab1, tab2, tab3 = st.tabs(["1x Scale", "10x Scale", "100x Scale"])

with tab1:
    st.dataframe(benchmark_1x)
    st.write("Recommendation for 1x Scale: Parquet + Snappy")
    st.write("Fastest read speed (0.067 sec).")
    st.write("Reasonable write speed (0.23 sec, 5× faster than CSV).")
    st.write("Good compression (~65% smaller than CSV).")
with tab2:
    st.dataframe(benchmark_10x)
    st.write("Recommendation for 10x Scale: Parquet + Snappy")
    st.write("Good balance between compression and speed.")
    st.write("60% smaller file than CSV.")
    st.write("Second fastest read time, only slightly slower than Parquet + Gzip but a perfect combination of all three metrics.")
with tab3:
    st.dataframe(benchmark_100x)
    st.write("Recommendation for 100x Scale: Parquet + Gzip")
    st.write("Best tradeoff between compression and read speed.")
    st.write("75% smaller than CSV (758MB vs. 2.88GB).")
    st.write("Read time is 6× faster than CSV.")

# Section B: Stock Market Analysis
st.subheader("📈 Stock Price Analysis")
chart_type = st.radio("Choose chart type:", ["Line Chart", "Candlestick Chart"], horizontal=True)

fig = go.Figure()
if chart_type == "Line Chart":
    fig = px.line(stock_df, x="date", y="close", title=f"{selected_stock} Closing Prices")
else:
    fig.add_trace(go.Candlestick(x=stock_df['date'],
                                 open=stock_df['open'],
                                 high=stock_df['high'],
                                 low=stock_df['low'],
                                 close=stock_df['close'],
                                 name='Candlestick'))
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Section C: Predictions (XGBoost & Random Forest)
st.subheader("🤖 Stock Price Predictions")
@st.cache
def load_models():
    xgb_model = joblib.load("xgb_model.pkl")
    return xgb_model

xgb_model = load_models()
latest_features = df_selected[['50_MA', '200_MA', 'RSI', 'MACD']].iloc[-1:].values
import numpy as np

# Ensure the features match the training shape
latest_features = np.array(latest_features).reshape(1, -1)

xgb_pred = xgb_model.predict(latest_features)

print(f"Expected Features: {xgb_model.n_features_in_}")
print(f"Provided Features: {latest_features.shape[1]}")

st.subheader("Stock Price Predictions")
st.write(f"**XGBoost Prediction:** {xgb_pred[0]:.2f}")
