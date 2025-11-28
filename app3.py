import streamlit as st
import pandas as pd
import numpy as np
import requests
import math

st.set_page_config(page_title="USD/INR Options Suite", layout="wide", page_icon="💹")

# =======================================
#          CUSTOM NORMAL FUNCTIONS
# =======================================
def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

# =======================================
#       GARMAN–KOHLHAGEN OPTION MODEL
# =======================================
def garman_kohlhagen(S, K, r_d, r_f, vol, T, option_type="CALL"):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type == "CALL":
        return S * np.exp(-r_f * T) * norm_cdf(d1) - K * np.exp(-r_d * T) * norm_cdf(d2)
    else:
        return K * np.exp(-r_d * T) * norm_cdf(-d2) - S * np.exp(-r_f * T) * norm_cdf(-d1)

# =======================================
# =======================================
#     REAL-TIME USD/INR PRICE FETCH
# =======================================
def get_realtime_usdinr():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=INR"
        data = requests.get(url).json()
        return round(data["rates"]["INR"], 4)
    except:
        return None

# =======================================
#     DUMMY USD/INR OPTION CHAIN DATA
# =======================================
def get_usdinr_option_chain():
    strikes = np.arange(80, 92, 0.5)
    data = {
        "Strike": strikes,
        "Call OI": np.random.randint(1000, 9000, len(strikes)),
        "Put OI": np.random.randint(1000, 9000, len(strikes)),
        "Call LTP": np.round(np.random.uniform(0.1, 1.5, len(strikes)), 2),
        "Put LTP": np.round(np.random.uniform(0.1, 1.5, len(strikes)), 2),
    }
    return pd.DataFrame(data)

# =======================================
#                 UI HEADER
# =======================================
st.markdown("""
<div style='text-align:center; padding: 20px;'>
    <h1 style='color:#0A81F5;'>💹 USD/INR Forex Options Dashboard</h1>
    <h3>Live Option Chain • Option Pricing • Moneyness Visualization</h3>
</div>
""", unsafe_allow_html=True)

# Tabs for clean navigation
tab1, tab2 = st.tabs(["📘 USD/INR Option Chain", "🧮 Option Calculator"])

# =======================================
#            OPTION CHAIN TAB
# =======================================
with tab1:
    st.subheader("📘 USD/INR Option Chain View")

    S_chain = st.number_input("Spot Price (for Moneyness)", value=83.00)
    df = get_usdinr_option_chain()

    # Define moneyness
    df["Call Moneyness"] = df["Strike"].apply(lambda k: "ITM" if S_chain > k else ("OTM" if S_chain < k else "ATM"))
    df["Put Moneyness"]  = df["Strike"].apply(lambda k: "ITM" if S_chain < k else ("OTM" if S_chain > k else "ATM"))

    # Row color styling
    def color_rows(row):
        m = row["Call Moneyness"]
        if m == "ITM":
            return ["background-color:#8BC34A; color:black"] * len(row)
        elif m == "ATM":
            return ["background-color:#FFEB3B; color:black"] * len(row)
        else:
            return ["background-color:#FF5722; color:white"] * len(row)

    st.dataframe(df.style.apply(color_rows, axis=1), use_container_width=True)

# =======================================
#            OPTION CALCULATOR TAB
# =======================================
with tab2:
    st.subheader("🧮 FX Option Pricing (Garman–Kohlhagen)")

    col1, col2, col3 = st.columns(3)

    with col1:
        F = st.number_input("Future Price (USD/INR)", value=83.20)
        S = st.number_input("Spot Price", value=83.00)
        K = st.number_input("Strike Price", value=83.00)

    with col2:
        r_d = st.number_input("Domestic Interest Rate (%)", value=6.5) / 100
        r_f = st.number_input("Foreign Interest Rate (%)", value=5.0) / 100
        vol = st.number_input("Volatility (%)", value=4.0) / 100

    with col3:
        T = st.number_input("Time to Expiry (Years)", value=0.0833)
        option_type = st.selectbox("Option Type", ["CALL", "PUT"])

    st.write(f"### Selected Strike: **{K}**")
    st.write(f"### Future Price: **{F}**")

    if st.button("Calculate Option Price", use_container_width=True):
        price = garman_kohlhagen(S, K, r_d, r_f, vol, T, option_type)
        st.success(f"### 🧾 Option Price: **{price:.4f}**")

# Footer
st.markdown("""
<hr>
<div style='text-align:center;'>
    <p>Built for USD/INR Forex Traders | Streamlit App</p>
</div>
""", unsafe_allow_html=True)
