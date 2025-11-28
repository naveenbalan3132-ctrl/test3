"""
Streamlit app: USD/INR Option Chain + Option Calculator
Files: single-file app (this file)

How to use / deploy
1) Create a GitHub repo and add this file as `streamlit_app.py` at the repo root.
2) Add `requirements.txt` with the packages listed below.
3) Push to GitHub. Go to https://share.streamlit.io, connect your GitHub, choose the repo and the file `streamlit_app.py` to deploy.

requirements.txt (add to repo):
streamlit
pandas
numpy
scipy
requests

Notes
- The app includes an internal sample USD/INR option chain if you don't supply a data source.
- You can also provide a public URL that returns CSV or JSON with an option chain compatible column names (see UI hint in app).
- Pricing model: Garman-Kohlhagen (Black-Scholes for FX) is implemented along with greeks.

"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="USD/INR Option Chain & Calculator", layout="wide")

st.title("USD/INR Option Chain — Viewer & Option Calculator")
st.markdown("Small Streamlit app showing an option chain for USD/INR and an option calculator using the Garman-Kohlhagen model.")

# --------------------
# Utility: Garman-Kohlhagen
# --------------------

def gk_price(S, K, T, r_dom, r_for, sigma, option='call'):
    # S: spot (domestic currency per foreign unit) e.g. INR per USD
    # r_dom: domestic interest rate (decimal, continuous)
    # r_for: foreign interest rate (decimal, continuous)
    # T: time to expiry in years
    if T <= 0 or sigma <= 0:
        # intrinsic payoff
        if option == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    df_dom = np.exp(-r_dom * T)
    df_for = np.exp(-r_for * T)
    forward = S * df_for / df_dom
    # Black-76 form using forward F
    F = forward
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    if option == 'call':
        price = df_dom * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = df_dom * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return float(price)


def gk_greeks(S, K, T, r_dom, r_for, sigma, option='call'):
    # Returns dict of greeks: delta (domestic-quoted), gamma, vega, theta (per day), rho_dom, rho_for
    if T <= 0 or sigma <= 0:
        # At expiry greeks undefined — approximate
        delta = 1.0 if (option=='call' and S>K) else 0.0
        return {'price': gk_price(S,K,T,r_dom,r_for,sigma,option), 'delta': delta, 'gamma': 0.0, 'vega':0.0, 'theta':0.0, 'rho_dom':0.0, 'rho_for':0.0}
    df_dom = np.exp(-r_dom * T)
    df_for = np.exp(-r_for * T)
    F = S * df_for / df_dom
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    price = gk_price(S, K, T, r_dom, r_for, sigma, option)

    # Delta (domestic-quoted): dPrice/dS
    # For FX: delta = exp(-r_for * T) * N(d1) for call (Black-style)
    if option == 'call':
        delta = df_for * cdf_d1
    else:
        delta = -df_for * norm.cdf(-d1)
    # Gamma
    gamma = df_for * pdf_d1 / (S * vol_sqrt_T)
    # Vega (per 1 vol, not per 1%)
    vega = df_dom * S * df_for * pdf_d1 * np.sqrt(T) / S  # simplifies to df_dom * df_for * pdf_d1 * sqrt(T)
    # but more straightforward:
    vega = df_dom * F * pdf_d1 * np.sqrt(T)
    # Theta (per day)
    # Derivative wrt time of Black-76 price. We give per-calendar-day approximation
    # Use finite-difference small dt
    dt = 1.0 / 365.0
    price_plus = gk_price(S, K, max(T - dt, 0.0), r_dom, r_for, sigma, option)
    theta = (price_plus - price)  # change per day (negative typically)
    # Rho wrt domestic rate (r_dom): partial derivative
    # Finite difference
    dr = 1e-4
    price_r_plus = gk_price(S, K, T, r_dom + dr, r_for, sigma, option)
    rho_dom = (price_r_plus - price) / dr
    # Rho wrt foreign rate
    price_rf_plus = gk_price(S, K, T, r_dom, r_for + dr, sigma, option)
    rho_for = (price_rf_plus - price) / dr

    return {'price': price, 'delta': float(delta), 'gamma': float(gamma), 'vega': float(vega), 'theta': float(theta), 'rho_dom': float(rho_dom), 'rho_for': float(rho_for)}

# --------------------
# Sample option chain generator (if no data source provided)
# --------------------

def make_sample_chain(spot=83.50, today=None):
    if today is None:
        today = datetime.utcnow().date()
    expiries = [today + timedelta(days=7), today + timedelta(days=30), today + timedelta(days=90)]
    strikes = np.arange(int(spot*0.8), int(spot*1.2)+1, 1)
    rows = []
    for exp in expiries:
        T = (exp - today).days / 365.0
        for K in strikes:
            # synthetic implied vol: ATM low, wings higher
            moneyness = abs((K - spot) / spot)
            iv = 0.08 + 0.25 * moneyness + 0.02 * np.random.rand()
            call_price = gk_price(spot, K, T, 0.06, 0.02, iv, 'call')
            put_price = gk_price(spot, K, T, 0.06, 0.02, iv, 'put')
            rows.append({'expiry': exp.isoformat(), 'strike': K, 'iv': round(iv,4), 'call_mid': round(call_price,4), 'put_mid': round(put_price,4)})
    df = pd.DataFrame(rows)
    return df

# --------------------
# Sidebar controls / data loading
# --------------------

st.sidebar.header("Data source & settings")
data_source = st.sidebar.radio("Load option chain from:", ['Sample internal chain', 'Upload CSV', 'Fetch from URL'])

option_chain_df = None

if data_source == 'Sample internal chain':
    spot_default = 83.50
    spot = st.sidebar.number_input('Spot INR per USD (S)', value=float(spot_default), step=0.01, format="%.4f")
    option_chain_df = make_sample_chain(spot=spot)
elif data_source == 'Upload CSV':
    uploaded = st.sidebar.file_uploader('Upload CSV (columns: expiry,strike,iv,call_mid,put_mid)', type=['csv'])
    if uploaded is not None:
        option_chain_df = pd.read_csv(uploaded)
elif data_source == 'Fetch from URL':
    url = st.sidebar.text_input('Public URL (returns CSV or JSON)')
    if url:
        try:
            resp = requests.get(url, timeout=10)
            text = resp.text
            # try CSV then JSON
            try:
                option_chain_df = pd.read_csv(StringIO(text))
            except Exception:
                option_chain_df = pd.read_json(StringIO(text))
        except Exception as e:
            st.sidebar.error(f"Failed to fetch: {e}")

# --------------------
# Display option chain
# --------------------

st.header("Option chain — USD/INR")
if option_chain_df is None:
    st.info('No option chain loaded yet. Select a data source on the left or use the sample chain.')
else:
    # Basic normalization
    df = option_chain_df.copy()
    # ensure columns
    expected = ['expiry','strike','iv','call_mid','put_mid']
    for c in expected:
        if c not in df.columns:
            # try lower/upper variants
            if c.upper() in df.columns:
                df[c] = df[c.upper()]
            elif c.capitalize() in df.columns:
                df[c] = df[c.capitalize()]
            else:
                st.warning(f'Column `{c}` not found in supplied data — sample data has these columns.')
    # types
    if 'expiry' in df.columns:
        df['expiry'] = pd.to_datetime(df['expiry']).dt.date
    if 'strike' in df.columns:
        df['strike'] = pd.to_numeric(df['strike'])

    # Filters
    cols = st.columns([1,1,1,1])
    with cols[0]:
        expiry_sel = st.selectbox('Expiry', options=sorted(df['expiry'].unique()), index=0)
    with cols[1]:
        strike_min = int(df['strike'].min())
        strike_max = int(df['strike'].max())
        strike_sel = st.slider('Strike range', strike_min, strike_max, (strike_min, strike_max))
    with cols[2]:
        iv_max = float(df['iv'].max()) if 'iv' in df.columns else 0.5
        iv_thresh = st.slider('Max IV', 0.0, max(0.5, iv_max), float(iv_max))
    with cols[3]:
        show_plot = st.checkbox('Show IV / premiums chart', value=True)

    df_view = df[(df['expiry']==expiry_sel) & (df['strike']>=strike_sel[0]) & (df['strike']<=strike_sel[1])]
    st.dataframe(df_view.reset_index(drop=True))

    if show_plot:
        try:
            chart_df = df_view[['strike','iv','call_mid','put_mid']].set_index('strike')
            st.line_chart(chart_df)
        except Exception:
            pass

# --------------------
# Option Calculator
# --------------------

st.header('Option calculator (Garman-Kohlhagen for FX)')
col1, col2 = st.columns(2)
with col1:
    S = st.number_input('Spot (INR per USD)', value=83.50, format="%.6f")
    K = st.number_input('Strike', value=83.0, format="%.6f")
    expiry_dt = st.date_input('Expiry date', value=(datetime.utcnow().date() + timedelta(days=30)))
    option_type = st.selectbox('Option type', ['call','put'])
with col2:
    r_dom = st.number_input('Domestic interest rate (INR) — annual continuous, e.g. 0.06', value=0.06, format="%.6f")
    r_for = st.number_input('Foreign interest rate (USD) — annual continuous, e.g. 0.02', value=0.02, format="%.6f")
    sigma = st.number_input('Implied volatility (annual, decimal) e.g. 0.12', value=0.12, format="%.6f")

T = max((expiry_dt - datetime.utcnow().date()).days / 365.0, 0.0)

if st.button('Calculate'):
    res = gk_greeks(S, K, T, r_dom, r_for, sigma, option_type)
    st.subheader('Results')
    st.write(f"Price ({option_type}): {res['price']:.6f} INR")
    cols = st.columns(3)
    cols[0].metric('Delta', f"{res['delta']:.6f}")
    cols[1].metric('Gamma', f"{res['gamma']:.6e}")
    cols[2].metric('Vega', f"{res['vega']:.6f}")
    cols2 = st.columns(3)
    cols2[0].metric('Theta (per day)', f"{res['theta']:.6f}")
    cols2[1].metric('Rho (domestic)', f"{res['rho_dom']:.6f}")
    cols2[2].metric('Rho (foreign)', f"{res['rho_for']:.6f}")

    st.markdown('---')
    st.markdown('**Interpretation hints**: Delta is expressed in domestic currency terms: approximate change in option price (INR) per 1 unit move in spot (INR per USD). Vega is change per 1.0 absolute vol (not per 1 vol point). Theta shown is approximate change per calendar day using a 1/365 step. Rho_dom is sensitivity to domestic interest rate; rho_for to foreign rate.')

st.markdown('\n---\n')
st.caption('This app is provided as sample code. For production use, connect to a reliable market data API for USD/INR option chains, add error handling and caching, and secure API keys as necessary.')
