import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime

st.set_page_config(page_title="USD/INR Options Suite", layout="wide", page_icon="💹")

# --------------------------- Utility: Normal functions (no SciPy) ---------------------------
def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def norm_cdf(x):
    # Use math.erf for stability
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# --------------------------- Garman-Kohlhagen pricing (FX Black-Scholes) ---------------------------
def garman_kohlhagen(S, K, r_d, r_f, vol, T, option_type="CALL"):
    if T <= 0 or vol <= 0:
        return max(0.0, S - K) if option_type == "CALL" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if option_type == "CALL":
        return S * math.exp(-r_f * T) * norm_cdf(d1) - K * math.exp(-r_d * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r_d * T) * norm_cdf(-d2) - S * math.exp(-r_f * T) * norm_cdf(-d1)

# --------------------------- Real-time USD/INR fetch (free) ---------------------------
def get_realtime_usdinr():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=INR"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return float(round(data["rates"]["INR"], 4))
    except Exception:
        return None

# --------------------------- Dummy option chain (replaceable) ---------------------------
def get_usdinr_option_chain():
    # Placeholder synthetic chain — replace with a market API when available
    strikes = np.arange(78.0, 89.5, 0.5)
    data = {
        "Strike": strikes,
        "Call OI": np.random.randint(1000, 9000, len(strikes)),
        "Put OI": np.random.randint(1000, 9000, len(strikes)),
        "Call LTP": np.round(np.random.uniform(0.05, 2.00, len(strikes)), 3),
        "Put LTP": np.round(np.random.uniform(0.05, 2.00, len(strikes)), 3),
    }
    return pd.DataFrame(data)

# --------------------------- Header & live ticker ---------------------------
st.markdown(
    "<div style='text-align:center; padding: 12px 0;'>"
    "<h1 style='color:#0A81F5; margin:0;'>💹 USD/INR Options Dashboard</h1>"
    "<div style='color:#555;'>Live option chain • option calculator • moneyness</div>"
    "</div>",
    unsafe_allow_html=True,
)

# Try to auto-refresh every 10s if the optional package is present
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, key="autorefresh")
    auto_ref_msg = "(autorefresh every 10s)"
except Exception:
    auto_ref_msg = "(click 'Refresh price')"

# Fetch live spot
spot_live = get_realtime_usdinr()
col_top = st.columns([1, 3, 1])
with col_top[0]:
    if spot_live is not None:
        st.metric(label="🇺🇸🇮🇳 Live USD/INR", value=f"{spot_live:.4f}", delta=None)
    else:
        st.metric(label="🇺🇸🇮🇳 Live USD/INR", value="N/A")
with col_top[2]:
    if spot_live is None:
        if st.button("Refresh price"):
            spot_live = get_realtime_usdinr()

st.markdown(f"**Data refresh:** {auto_ref_msg}")

# --------------------------- Tabs ---------------------------
tab_chain, tab_calc = st.tabs(["📘 Option Chain", "🧮 Option Calculator"])

# --------------------------- Option Chain tab ---------------------------
with tab_chain:
    st.subheader("USD/INR Option Chain (synthetic)")
    S_chain = st.number_input("Spot Price (for moneyness)", value=spot_live if spot_live else 83.00, format="%.4f")

    df = get_usdinr_option_chain()
    # compute moneyness for call and put
    df["Call Moneyness"] = df["Strike"].apply(lambda k: "ITM" if S_chain > k else ("OTM" if S_chain < k else "ATM"))
    df["Put Moneyness"]  = df["Strike"].apply(lambda k: "ITM" if S_chain < k else ("OTM" if S_chain > k else "ATM"))

    # highlight rows by call moneyness and also color call/put columns separately
    def row_style(row):
        m = row["Call Moneyness"]
        if m == "ITM":
            return ["background-color:#e8f5e9; color:black"] * len(row)
        elif m == "ATM":
            return ["background-color:#fffde7; color:black"] * len(row)
        else:
            return ["background-color:#ffebee; color:black"] * len(row)

    styled = df.style.apply(row_style, axis=1)
    # additionally highlight the Call LTP and Put LTP columns by their own moneyness
    def highlight_call(val):
        return "font-weight: bold" if val >= 0 else ""

    st.dataframe(styled, use_container_width=True, height=420)

    # Allow CSV export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download option chain CSV", data=csv, file_name=f"usdinr_chain_{datetime.now().date()}.csv")

# --------------------------- Option Calculator tab ---------------------------
with tab_calc:
    st.subheader("FX Option Calculator (Garman–Kohlhagen)")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        # auto-fill spot and future with live price if available
        S = st.number_input("Spot Price (USD/INR)", value=spot_live if spot_live else 83.00, format="%.4f")
        F = st.number_input("Future Price (USD/INR)", value=spot_live if spot_live else 83.20, format="%.4f")
        K = st.number_input("Strike Price", value=round(S, 2), format="%.4f")
    with col2:
        r_d = st.number_input("Domestic Interest Rate (%)", value=6.5, format="%.3f") / 100.0
        r_f = st.number_input("Foreign Interest Rate (%)", value=2.5, format="%.3f") / 100.0
        vol = st.number_input("Volatility (annual %)", value=8.0, format="%.4f") / 100.0
    with col3:
        T = st.number_input("Time to Expiry (years)", value=0.0833, format="%.6f")
        option_type = st.selectbox("Option Type", ["CALL", "PUT"]) 
        qty = st.number_input("Quantity (positive = long)", value=1, step=1)

    st.markdown("---")
    st.write(f"**Selected Strike:** {K} — **Future Price:** {F}")

    if st.button("Calculate Price & Greeks", use_container_width=True):
        price = garman_kohlhagen(S, K, r_d, r_f, vol, T, option_type)
        # rudimentary Greeks approx
        eps = 1e-4
        price_up = garman_kohlhagen(S + eps, K, r_d, r_f, vol, T, option_type)
        delta = (price_up - price) / eps
        vega = (garman_kohlhagen(S, K, r_d, r_f, vol + 0.01, T, option_type) - price) / 0.01

        st.success(f"Option Price: {price:.6f} INR")
        cola, colb, colc = st.columns(3)
        cola.metric("Delta", f"{delta:.6f}")
        colb.metric("Vega", f"{vega:.6f}")
        colc.metric("PnL (qty) at expiry (spot=F)", f"{(max(0, (F - K) if option_type=='CALL' else (K - F)) - price) * qty:.6f} INR")

    # payoff chart
    st.markdown("### Payoff at expiry (per USD)")
    S_range = np.linspace(max(0.5 * S, K - 5), max(1.5 * S, K + 5), 200)
    if option_type == "CALL":
        payoff = np.maximum(S_range - K, 0) - price
    else:
        payoff = np.maximum(K - S_range, 0) - price
    payoff = payoff * qty

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(S_range, payoff)
    ax.axhline(0, color='black')
    ax.set_xlabel('Spot at expiry (INR)')
    ax.set_ylabel('P&L (INR)')
    ax.set_title('Payoff at Expiry')
    st.pyplot(fig)

# --------------------------- Footer ---------------------------
st.markdown("---")
st.caption("This tool uses a synthetic option chain. Replace the chain fetch with a market data API for live option chains and implied volatilities.")
