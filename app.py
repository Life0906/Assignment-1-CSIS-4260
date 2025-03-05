import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestRegressor

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
st.line_chart(df_selected[['MACD']])

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
