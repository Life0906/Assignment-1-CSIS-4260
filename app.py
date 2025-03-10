import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load precomputed benchmarking results
benchmark_1x = pd.read_csv("benchmark_results.csv")
benchmark_10x = pd.read_csv("benchmark_results_10x.csv")
benchmark_100x = pd.read_csv("benchmark_results_100x.csv")

# Load dataset
def load_data():
    df = pd.read_parquet("data.parquet")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Sidebar: Select Company
companies = df['name'].unique()
selected_company = st.sidebar.selectbox("Select a Stock:", companies)
df_selected = df[df['name'] == selected_company]

# Sidebar: Chart Type Selection
chart_type = st.sidebar.radio("Select Chart Type:", ["Line Chart", "Candlestick Chart"])

# Compute Technical Indicators
df_selected['50_MA'] = df_selected['close'].rolling(window=50).mean()
df_selected['200_MA'] = df_selected['close'].rolling(window=200).mean()
df_selected['RSI'] = 100 - (100 / (1 + df_selected['close'].pct_change().rolling(14).mean()))
df_selected['MACD'] = df_selected['close'].ewm(span=12, adjust=False).mean() - df_selected['close'].ewm(span=26, adjust=False).mean()
df_selected['Signal_Line'] = df_selected['MACD'].ewm(span=9, adjust=False).mean()

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

# Plot Charts
st.title(f"Stock Analysis Dashboard for {selected_company}")
if chart_type == "Line Chart":
    fig = px.line(df_selected, x='date', y='close', title=f"{selected_company} Closing Prices")
else:
    fig = go.Figure(data=[go.Candlestick(
        x=df_selected['date'],
        open=df_selected['open'],
        high=df_selected['high'],
        low=df_selected['low'],
        close=df_selected['close']
    )])
    fig.update_layout(title=f"{selected_company} Candlestick Chart")
st.plotly_chart(fig)

# Display Moving Averages, RSI, MACD
st.subheader("Technical Indicators")
st.line_chart(df_selected[['50_MA', '200_MA']])
st.line_chart(df_selected[['RSI']])
st.line_chart(df_selected[['MACD','Signal_Line']])

# Load & Display Predictions (Assume Models Are Pretrained & Saved)
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
