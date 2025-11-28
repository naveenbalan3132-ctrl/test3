# Streamlit USD/INR Option Chain + Option Calculator
# --------------------------------------------------
# Single-file Streamlit app. Put this file as `app.py` in the root of a GitHub repo
# and deploy to https://share.streamlit.io by selecting the repo and branch.
#
# Requirements (requirements.txt):
# streamlit
# numpy
# pandas
# scipy
# matplotlib
#
# Quick deploy steps:
# 1. Create a GitHub repo and add this file as `app.py`.
# 2. Add a requirements.txt with the packages listed above.
# 3. Go to https://share.streamlit.io, connect your GitHub, click "New app",
#    choose the repo, branch and set "Entry file" to `app.py`.
# 4. Launch — Streamlit Cloud will install the requirements and run the app.
#
# Notes:
# - This app generates a synthetic option chain for the USD/INR pair using the
#   Garman-Kohlhagen (Black-Scholes for FX) model. If you want live spot or
#   market-implied vols, integrate a market data API and replace the default inputs.
# - The app also contains a detailed option calculator (price, Greeks, breakeven,
#   payoff diagrams, export CSV).

import streamlit as st
import numpy as np
import pandas as pd
# Removed SciPy — using custom normal PDF/CDF
import math
from datetime import datetime, date
import math
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="USD/INR Option Chain & Calculator", layout="wide")

# --------- Financial math: Garman-Kohlhagen (Black-Scholes for FX) ---------

# --- Custom Normal Distribution Functions (No SciPy Needed) ---
def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

# Abramowitz-Stegun approximation for normal CDF
def norm_cdf(x):
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)
 # --------- Financial math: Garman-Kohlhagen (Black–Scholes for FX) ---------

def year_fraction(start_date, end_date):
    # Actual/365 simple
    return max((end_date - start_date).days / 365.0, 1e-10)


def gk_price_call(S, K, T, sigma, rd, rf):
    # S: spot price (domestic currency per unit foreign — e.g., INR per USD)
    # K: strike
    # T: time to expiry in years
    # sigma: volatility (annual)
    # rd: domestic interest rate (annual, decimal)
    # rf: foreign interest rate (annual, decimal)
    if T <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df_dom = math.exp(-rd * T)
    df_for = math.exp(-rf * T)
    call = S * df_for * norm.cdf(d1) - K * df_dom * norm.cdf(d2)
    return call


def gk_price_put(S, K, T, sigma, rd, rf):
    if T <= 0:
        return max(K - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df_dom = math.exp(-rd * T)
    df_for = math.exp(-rf * T)
    put = K * df_dom * norm.cdf(-d2) - S * df_for * norm.cdf(-d1)
    return put


def gk_greeks(S, K, T, sigma, rd, rf, option_type="call"):
    # returns dict of price and Greeks: delta, gamma, vega, theta, rho_dom, rho_for
    if T <= 0:
        price = max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        return {"price": price, "delta": 0.0, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho_dom": 0.0, "rho_for": 0.0}
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df_dom = math.exp(-rd * T)
    df_for = math.exp(-rf * T)
    pdf_d1 = norm.pdf(d1)

    if option_type == 'call':
        price = S * df_for * norm.cdf(d1) - K * df_dom * norm.cdf(d2)
        delta = df_for * norm.cdf(d1)
        theta = (-S * df_for * pdf_d1 * sigma / (2 * sqrtT)
                 - rd * K * df_dom * norm.cdf(d2)
                 + rf * S * df_for * norm.cdf(d1))
        rho_dom = -T * K * df_dom * norm.cdf(d2)
        rho_for = T * S * df_for * norm.cdf(d1)
    else:
        price = K * df_dom * norm.cdf(-d2) - S * df_for * norm.cdf(-d1)
        delta = -df_for * norm.cdf(-d1)
        theta = (-S * df_for * pdf_d1 * sigma / (2 * sqrtT)
                 + rd * K * df_dom * norm.cdf(-d2)
                 - rf * S * df_for * norm.cdf(-d1))
        rho_dom = T * K * df_dom * norm.cdf(-d2)
        rho_for = -T * S * df_for * norm.cdf(-d1)

    gamma = df_for * pdf_d1 / (S * sigma * sqrtT)
    vega = S * df_for * pdf_d1 * sqrtT

    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho_dom": rho_dom, "rho_for": rho_for}

# --------------------------- Streamlit UI ---------------------------------

st.title("USD/INR Option Chain & Option Calculator")
st.markdown(
    "Generate a synthetic option chain for USD/INR and compute option prices + Greeks using the Garman–Kohlhagen model (Black–Scholes for FX)."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Market inputs")
    spot = st.number_input("Spot (INR per USD)", value=83.50, format="%.6f")
    expiry = st.date_input("Expiry date", value=date.today())
    vol_annual = st.number_input("Volatility (annual %, sigma)", value=8.0, min_value=0.01, format="%.4f")
    dom_rate = st.number_input("Domestic rate (INR, annual %)", value=6.5, format="%.3f")
    for_rate = st.number_input("Foreign rate (USD, annual %)", value=2.5, format="%.3f")
    spot_fetch = st.checkbox("Auto-fetch spot (not implemented)", value=False, help="Integrate API to fetch live spot; currently placeholder.")

with col2:
    st.subheader("Option chain settings")
    strikes_around = st.number_input("Strikes around spot (count each side)", value=10, min_value=1)
    strike_step = st.number_input("Strike step (INR)", value=0.25, min_value=0.01, format="%.4f")
    strike_min = st.number_input("Min strike (leave 0 to auto)", value=0.0, format="%.4f")
    strike_max = st.number_input("Max strike (leave 0 to auto)", value=0.0, format="%.4f")
    show_greeks = st.checkbox("Show Greeks", value=True)
    export_csv = st.checkbox("Show export CSV button", value=True)

# compute time to expiry
today = date.today()
T = year_fraction(today, expiry)
sigma = vol_annual / 100.0
rd = dom_rate / 100.0
rf = for_rate / 100.0

# generate strikes
if strike_min <= 0 or strike_max <= 0:
    center = spot
    strikes = []
    # create symmetric strikes around spot
    for i in range(-strikes_around, strikes_around + 1):
        strikes.append(round(center + i * strike_step, 6))
else:
    strikes = list(np.arange(strike_min, strike_max + 0.0000001, strike_step))

strikes = sorted([s for s in strikes if s > 0])

# build option chain
chain_rows = []
for K in strikes:
    call_price = gk_price_call(spot, K, T, sigma, rd, rf)
    put_price = gk_price_put(spot, K, T, sigma, rd, rf)
    call_greeks = gk_greeks(spot, K, T, sigma, rd, rf, option_type='call')
    put_greeks = gk_greeks(spot, K, T, sigma, rd, rf, option_type='put')
    chain_rows.append({
        'strike': K,
        'call': call_price,
        'put': put_price,
        'call_delta': call_greeks['delta'],
        'put_delta': put_greeks['delta'],
        'call_gamma': call_greeks['gamma'],
        'put_gamma': put_greeks['gamma'],
        'call_vega': call_greeks['vega'],
        'put_vega': put_greeks['vega'],
        'call_theta': call_greeks['theta'],
        'put_theta': put_greeks['theta'],
    })

chain_df = pd.DataFrame(chain_rows)
chain_df = chain_df[['strike', 'call', 'put', 'call_delta', 'put_delta', 'call_gamma', 'put_gamma', 'call_vega', 'put_vega', 'call_theta', 'put_theta']]

st.subheader("Synthetic Option Chain (USD/INR)")
st.markdown(f"Spot = **{spot:.6f}** INR/USD — Expiry = **{expiry.isoformat()}** — T = **{T:.6f} yrs** — Vol = **{vol_annual:.2f}%**")

# show table with download option
st.dataframe(chain_df.style.format({
    'strike': '{:.4f}', 'call': '{:.4f}', 'put': '{:.4f}',
    'call_delta': '{:.4f}', 'put_delta': '{:.4f}',
    'call_gamma': '{:.6f}', 'put_gamma': '{:.6f}',
    'call_vega': '{:.4f}', 'put_vega': '{:.4f}',
    'call_theta': '{:.4f}', 'put_theta': '{:.4f}'
}), height=400)

if export_csv:
    csv_bytes = chain_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download option chain CSV", data=csv_bytes, file_name=f"usdinr_option_chain_{expiry.isoformat()}.csv")

# small chart: call & put prices vs strike
fig, ax = plt.subplots()
ax.plot(chain_df['strike'], chain_df['call'], label='Call')
ax.plot(chain_df['strike'], chain_df['put'], label='Put')
ax.set_xlabel('Strike (INR)')
ax.set_ylabel('Price (INR)')
ax.set_title('Option prices vs Strike')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# -------------------- Option calculator (single-leg) ---------------------
st.subheader("Option Calculator — Single Leg")
colA, colB = st.columns(2)
with colA:
    opt_type = st.selectbox('Option type', ['call', 'put'])
    K0 = st.number_input('Strike', value=round(spot, 2), format='%.4f')
    qty = st.number_input('Quantity (positive for long, negative for short)', value=1, format='%.0f')
    prem_adj = st.number_input('Premium adjustment (market premium override). Leave 0 to use model price.', value=0.0, format='%.6f')

with colB:
    vol0 = st.number_input('Volatility (annual %)', value=vol_annual, format='%.4f')
    rd0 = st.number_input('Domestic rate (annual %)', value=dom_rate, format='%.4f')
    rf0 = st.number_input('Foreign rate (annual %)', value=for_rate, format='%.4f')
    expiry0 = st.date_input('Expiry (for this leg)', value=expiry, key='expiry0')

T0 = year_fraction(date.today(), expiry0)
sigma0 = vol0 / 100.0
rd0 = rd0 / 100.0
rf0 = rf0 / 100.0

model_price = gk_price_call(spot, K0, T0, sigma0, rd0, rf0) if opt_type == 'call' else gk_price_put(spot, K0, T0, sigma0, rd0, rf0)
price_used = model_price if abs(prem_adj) < 1e-12 else prem_adj

greeks = gk_greeks(spot, K0, T0, sigma0, rd0, rf0, option_type=opt_type)

st.markdown("**Result**")
res_col1, res_col2 = st.columns(2)
with res_col1:
    st.write(f"Model price: {model_price:.6f} INR")
    st.write(f"Price used: {price_used:.6f} INR (qty {qty})")
    st.write(f"Premium (cash): {price_used * qty:.6f} INR")
    st.write(f"Delta (per contract): {greeks['delta']:.6f}")
    st.write(f"Gamma: {greeks['gamma']:.8f}")
with res_col2:
    st.write(f"Vega: {greeks['vega']:.6f}")
    st.write(f"Theta: {greeks['theta']:.6f} (per year)")
    st.write(f"Rho (domestic): {greeks['rho_dom']:.6f}")
    st.write(f"Rho (foreign): {greeks['rho_for']:.6f}")

# payoff diagram
st.subheader('Payoff & P&L diagram (at expiry)')
S_range = np.linspace(max(0.5 * spot, strikes[0] * 0.5), max(1.5 * spot, strikes[-1] * 1.5), 200)
if opt_type == 'call':
    payoff = np.maximum(S_range - K0, 0) - price_used
else:
    payoff = np.maximum(K0 - S_range, 0) - price_used
payoff = payoff * qty

fig2, ax2 = plt.subplots()
ax2.plot(S_range, payoff)
ax2.axhline(0, color='black')
ax2.set_xlabel('Spot at expiry (INR)')
ax2.set_ylabel('P&L (INR)')
ax2.set_title('Payoff at Expiry')
ax2.grid(True)
st.pyplot(fig2)

# display breakeven
if qty != 0:
    if opt_type == 'call':
        breakeven = K0 + price_used / 1.0
    else:
        breakeven = K0 - price_used / 1.0
    st.write(f"Breakeven at expiry (per USD): {breakeven:.6f} INR")

# Export single-leg details
if st.button('Export leg details (CSV)'):
    buf = io.StringIO()
    df_leg = pd.DataFrame({
        'spot': [spot], 'strike': [K0], 'option_type': [opt_type], 'expiry': [expiry0.isoformat()],
        'price_used': [price_used], 'qty': [qty], 'delta': [greeks['delta']], 'gamma': [greeks['gamma']],
        'vega': [greeks['vega']], 'theta': [greeks['theta']], 'rho_dom': [greeks['rho_dom']], 'rho_for': [greeks['rho_for']]
    })
    df_leg.to_csv(buf, index=False)
    st.download_button('Download leg CSV', data=buf.getvalue().encode('utf-8'), file_name='option_leg.csv')

st.markdown('---')
st.caption('This tool produces *synthetic* option prices. For live tradable prices, connect to an exchange data API or broker API that provides market option chains and implied volatilities.')

# End of app
