import subprocess
import sys

def install_package(package, import_name=None):
    try:
        __import__(import_name or package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
required_libraries = {
    'yfinance': None,
    'numpy': None,
    'pandas': None,
    'matplotlib': None,
    'scipy': None,
    'torch': None,
    'streamlit': None,
    'plotly': None
}
for package, import_name in required_libraries.items():
    install_package(package, import_name)

# Imports
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go

# Top 50 BSE Companies
TOP_50_BSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'HINDUNILVR.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS',
    'SBIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'MARUTI.NS',
    'TITAN.NS', 'SUNPHARMA.NS', 'AXISBANK.NS', 'NTPC.NS', 'ONGC.NS',
    'POWERGRID.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'IOC.NS', 'COALINDIA.NS',
    'TATASTEEL.NS', 'JSWSTEEL.NS', 'WIPRO.NS', 'ADANIPORTS.NS', 'DRREDDY.NS',
    'CIPLA.NS', 'UPL.NS', 'BAJAJ-AUTO.NS', 'GRASIM.NS', 'TECHM.NS', 'SHREECEM.NS',
    'HEROMOTOCO.NS', 'INDUSINDBK.NS', 'DIVISLAB.NS', 'BPCL.NS', 'EICHERMOT.NS',
    'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'TATAMOTORS.NS', 'VEDL.NS',
    'M&M.NS', 'BRITANNIA.NS', 'HINDALCO.NS'
]

# Data fetching function
def fetch_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    if hist.empty:
        return None, None
    dates = hist.index.strftime('%Y-%m-%d').tolist()
    prices = hist['Close'].values
    return dates, prices

# Custom signal decomposition
def apply_custom_decomposition(prices):
    window_size = 10
    smoothed_prices = np.convolve(prices, np.ones(window_size)/window_size, mode='same')
    residuals = prices - smoothed_prices
    imfs = [smoothed_prices, residuals]
    analytic_signals = [hilbert(imf) for imf in imfs]
    amplitudes = [np.abs(signal) for signal in analytic_signals]
    return imfs, amplitudes

# TCN Model Definition
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=2)
        return self.fc(x)

# Prepare time series data
def prepare_data(prices, imfs, amplitudes, seq_length=30):
    X, y = [], []
    for i in range(len(prices) - seq_length):
        price_window = prices[i:i+seq_length].reshape(1, -1)
        imf_window = [imf[i:i+seq_length].reshape(1, -1) for imf in imfs]
        amp_window = [amp[i:i+seq_length].reshape(1, -1) for amp in amplitudes]
        combined_features = np.vstack([price_window, *imf_window, *amp_window])
        X.append(combined_features.T)
        y.append(prices[i+seq_length])
    return np.array(X), np.array(y)

# Train TCN model
def train_tcn(prices, imfs, amplitudes, epochs=50):
    X, y = prepare_data(prices, imfs, amplitudes)
    X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float32)
    num_features = X.shape[1]
    model = TCN(input_size=num_features, hidden_size=16, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.squeeze(), y)
        loss.backward()
        optimizer.step()
    return model

# Predict future prices
def predict_future_prices(model, prices, imfs, amplitudes, days_ahead=30):
    predictions = []
    seq_length = 30
    for _ in range(days_ahead):
        last_price_window = prices[-seq_length:].reshape(1, -1)
        imf_windows = [imf[-seq_length:].reshape(1, -1) for imf in imfs]
        amp_windows = [amp[-seq_length:].reshape(1, -1) for amp in amplitudes]
        combined = np.vstack([last_price_window, *imf_windows, *amp_windows])
        tensor_input = torch.tensor(combined.T, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        pred = model(tensor_input).item()
        predictions.append(pred)
        prices = np.append(prices, pred)
        imfs, amplitudes = apply_custom_decomposition(prices)
    return predictions

# Train models on top companies
def train_on_top_companies():
    models = {}
    performance = {}
    with st.spinner("Training models... This may take a few minutes"):
        progress = st.progress(0)
        for idx, symbol in enumerate(TOP_50_BSE):
            try:
                dates, prices = fetch_stock_data(symbol)
                if prices is None or len(prices) < 100:
                    continue
                imfs, amplitudes = apply_custom_decomposition(prices)
                model = train_tcn(prices, imfs, amplitudes)
                future_prices = predict_future_prices(model, prices, imfs, amplitudes, 30)
                predicted_return = (future_prices[-1] - prices[-1]) / prices[-1] * 100
                models[symbol] = model
                performance[symbol] = {
                    'current_price': prices[-1],
                    'predicted_price': future_prices[-1],
                    'predicted_return': predicted_return,
                    'imfs': imfs,
                    'amplitudes': amplitudes
                }
            except Exception as e:
                st.warning(f"⚠️ Error with {symbol}: {str(e)}")
            progress.progress((idx + 1) / len(TOP_50_BSE))
    return models, performance

# Recommend stocks based on user preferences
def recommend_stocks(performance, strategy, risk, horizon, budget):
    scores = []
    for symbol, data in performance.items():
        if data['predicted_return'] < 0:
            continue
        quantity_possible = int(budget / data['current_price'])
        if quantity_possible == 0:
            continue
        if strategy == 1:
            score = data['predicted_return'] * 2
        elif strategy == 2:
            score = 100 - abs(data['predicted_return'] - 20)
        elif strategy == 3:
            score = data['predicted_return'] + 50 - (data['predicted_return']**2)/100
        elif strategy == 4:
            score = data['predicted_return'] * 1.5
        else:
            score = data['predicted_return']
        if risk < 3:
            score -= abs(data['predicted_return']) * (3 - risk)
        else:
            score += abs(data['predicted_return']) * (risk - 2)
        scores.append((symbol, score))
    top_5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    return top_5

# Streamlit App UI
def streamlit_app():
    # Modernized Styling
    st.markdown(
        """
        <style>
        /* Background Gradient */
        .stApp {
            background-image: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: white !important;
        }

        /* Input Labels */
        .stTextInput > label,
        .stSelectbox > label,
        .stSlider > label,
        .stNumberInput > label,
        .stRadio > label,
        .stCheckbox > label {
            color: #ffffff !important;
            font-weight: 600;
            font-size: 16px;
        }

        /* Slider Track Color */
        .stSlider [data-baseweb="slider"] > div > div {
            background-color: #4CAF50 !important; /* Green */
        }

        /* Focus Ring */
        .stSlider [data-baseweb="slider"] > div > div:focus {
            box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.4);
        }

        /* Inputs */
        .stTextInput input,
        .stNumberInput input {
            background-color: #1e1e2f !important;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #444;
            padding: 8px;
        }

        .stSelectbox select {
            background-color: #1e1e2f !important;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #444;
        }

        /* Button Styling */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        .stButton button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }

        /* Table and Expander */
        .stTable {
            background-color: #1e1e2f !important;
            border-radius: 8px;
            overflow: hidden;
        }

        .streamlit-expanderHeader {
            background-color: #2c3e50;
            border-radius: 8px;
            padding: 10px;
            font-weight: bold;
            color: white;
        }

        /* Card-like container */
        .card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # Welcome Banner
    st.markdown("""
    <div class="card">
        <h3 style='text-align:center;'>📈 Welcome to Your AI-Powered Stock Advisor</h3>
        <p style='text-align:center;'>Enter your investment preferences to get smart stock recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 style='text-align: center;'>BSE Stock Recommendation System 📈</h1>", unsafe_allow_html=True)

    # User Inputs
    strategy = st.selectbox("Investment Strategy", ["High Growth", "Stable Returns", "Balanced", "Short-Term Gains", "Long-Term Investment"], index=0)
    strategy_map = {"High Growth": 1, "Stable Returns": 2, "Balanced": 3, "Short-Term Gains": 4, "Long-Term Investment": 5}
    risk = st.slider("Risk Appetite (1=Low, 5=High)", 1, 5, 3)
    horizon = st.slider("Investment Horizon (Months)", 1, 36, 6)
    budget = st.number_input("Investment Budget (INR)", min_value=1000.0, value=50000.0, step=1000.0)

    # Generate Recommendations
    if st.button("Recommend Stocks"):
        models, performance = train_on_top_companies()
        top_stocks = recommend_stocks(performance, strategy_map[strategy], risk, horizon, budget)
        if top_stocks:
            st.session_state.models = models
            st.session_state.performance = performance
            st.session_state.top_stocks = top_stocks
        else:
            st.error("No suitable stocks found within your budget and criteria.")
            return

    # Show Recommendations
    if "top_stocks" in st.session_state:
        top_stocks = st.session_state.top_stocks
        performance = st.session_state.performance
        models = st.session_state.models

        st.subheader("Top 5 Recommendations")
        st.table([
            {
                "Symbol": symbol.replace(".NS", ""),
                "Current Price": f"₹{performance[symbol]['current_price']:.2f}",
                "Predicted Price": f"₹{performance[symbol]['predicted_price']:.2f}",
                "Return %": f"{performance[symbol]['predicted_return']:.2f}%",
                "Qty Possible": int(budget / performance[symbol]['current_price'])
            }
            for symbol, _ in top_stocks
        ])

        selected_symbol = st.selectbox("Select a company to view its forecast:", [symbol for symbol, _ in top_stocks])

        # Fetch historical data again
        dates, prices = fetch_stock_data(selected_symbol)
        future = predict_future_prices(
            models[selected_symbol],
            prices,
            performance[selected_symbol]['imfs'],
            performance[selected_symbol]['amplitudes'],
            30
        )

        # Date handling
        historical_dates = dates[-60:]
        historical_prices = prices[-60:]
        future_dates = pd.date_range(start=pd.to_datetime(dates[-1]) + pd.Timedelta(days=1), periods=30).strftime('%Y-%m-%d')

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=future_dates, y=future, mode='lines+markers', name='Predicted', line=dict(dash='dash', color='red')))
        fig.update_layout(
            title=f"{selected_symbol.replace('.NS', '')} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template='plotly_dark',
            hovermode="x unified",
            height=500
        )

        st.markdown("### Detailed View")
        with st.expander(f"{selected_symbol.replace('.NS', '')} Stock Forecast Details", expanded=True):
            st.write(f"**Current Price:** ₹{performance[selected_symbol]['current_price']:.2f}")
            st.write(f"**Predicted Price (30 days):** ₹{performance[selected_symbol]['predicted_price']:.2f}")
            st.write(f"**Expected Return:** {performance[selected_symbol]['predicted_return']:.2f}%")

        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    streamlit_app()