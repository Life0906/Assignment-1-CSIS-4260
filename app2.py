import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load precomputed benchmarking results
benchmark_1x = pd.read_csv("benchmark_results.csv")
benchmark_10x = pd.read_csv("benchmark_results_10x.csv")
benchmark_100x = pd.read_csv("benchmark_results_100x.csv")

# Load stock data
stock_data = pd.read_parquet("scaled_dataset_1x_snappy.parquet")

# Streamlit App
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("Stock Market Dashboard")

# Sidebar for Stock Selection
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.selectbox("Choose a stock:", stock_data['symbol'].unique())

# Filter Data for Selected Stock
stock_df = stock_data[stock_data['symbol'] == selected_stock]

# Section A: Storage Benchmarking
st.subheader("📊 Storage Format Benchmarking")
tab1, tab2, tab3 = st.tabs(["1x Scale", "10x Scale", "100x Scale"])

with tab1:
    st.dataframe(benchmark_1x)
with tab2:
    st.dataframe(benchmark_10x)
with tab3:
    st.dataframe(benchmark_100x)

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
st.write("Predictions from XGBoost & Random Forest models will be shown here.")
