import streamlit as st
import pandas as pd
import numpy as np
import requests
import math

# ---------------------- FX OPTION CALCULATOR (Garman–Kohlhagen) ----------------------
def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def garman_kohlhagen(S, K, r_d, r_f, vol, T, option_type="CALL"):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type == "CALL":
        price = S * np.exp(-r_f * T) * norm_cdf(d1) - K * np.exp(-r_d * T) * norm_cdf(d2)
    else:
        price = K * np.exp(-r_d * T) * norm_cdf(-d2) - S * np.exp(-r_f * T) * norm_cdf(-d1)

    return price

# ---------------------- FETCH USD/INR OPTION CHAIN (Dummy Example) ----------------------
def get_usdinr_option_chain():
    # Placeholder example — replace with real data API
    strikes = np.arange(80, 92, 0.5)
    data = {
        "Strike": strikes,
        "Call OI": np.random.randint(1000, 9000, len(strikes)),
        "Put OI": np.random.randint(1000, 9000, len(strikes)),
        "Call LTP": np.round(np.random.uniform(0.1, 1.5, len(strikes)), 2),
        "Put LTP": np.round(np.random.uniform(0.1, 1.5, len(strikes)), 2),
    }
    return pd.DataFrame(data)

# ---------------------- STREAMLIT UI ----------------------
st.title("USD/INR Forex Option Tool")
st.write("Real-time USD/INR Option Chain + FX Option Calculator (Garman–Kohlhagen)")

menu = st.sidebar.selectbox("Select Feature", ["USD/INR Option Chain", "Option Calculator"])

# ---------------------- OPTION CHAIN SCREEN ----------------------
if menu == "USD/INR Option Chain":
    st.header("USD/INR Option Chain")
    df = get_usdinr_option_chain()
    st.dataframe(df)

# ---------------------- OPTION CALCULATOR SCREEN ----------------------
if menu == "Option Calculator":
    st.header("FX Option Calculator (Garman–Kohlhagen)")

    F = st.number_input("Future Price (USD/INR)", value=83.20)
    S = st.number_input("Spot Price (USD/INR)", value=83.00)
    K = st.number_input("Strike Price", value=83.00)
    r_d = st.number_input("Domestic Interest Rate (%)", value=6.5) / 100
    r_f = st.number_input("Foreign Interest Rate (%)", value=5.0) / 100
    vol = st.number_input("Volatility (%)", value=4.0) / 100
    T = st.number_input("Time to Expiry (Years)", value=0.0833)

    option_type = st.selectbox("Option Type", ["CALL", "PUT"])

    st.write(f"Selected Strike: {K}")
    st.write(f"Future Price: {F}")

    if st.button("Calculate Option Price"):
        price = garman_kohlhagen(S, K, r_d, r_f, vol, T, option_type)
        st.success(f"Option Price: {price:.4f}")

st.write("---")
st.caption("Deploy on Streamlit Cloud: https://share.streamlit.io")
