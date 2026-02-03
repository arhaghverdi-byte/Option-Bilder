import streamlit as st
import numpy as np
import plotly.graph_objects as go
import mibian
import pandas as pd

# ----------------------------------
# Session State
# ----------------------------------
if "legs" not in st.session_state:
    st.session_state.legs = []
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx = None

# ----------------------------------
# Utils
# ----------------------------------
def option_payoff(price, leg):
    k, p, q = leg['Strike'], leg['Premium'], leg['Qty']
    mult = 100 * abs(q)
    sign = np.sign(q)
    if leg['Type'] == 'Call':
        return sign * (np.maximum(price - k, 0) - p) * mult
    else:
        return sign * (np.maximum(k - price, 0) - p) * mult

def bs_greeks(S, leg, iv, r, days):
    bs = mibian.BS([float(S), leg['Strike'], r * 100, max(days,1)], volatility=iv)
    if leg['Type'] == 'Call':
        return bs.callDelta, bs.gamma, bs.callTheta, bs.vega
    else:
        return bs.putDelta, bs.gamma, bs.putTheta, bs.vega

# ----------------------------------
# Page Config + Style
# ----------------------------------
st.set_page_config(page_title="Option Strategy", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600&display=swap" rel="stylesheet">
<style>
* {direction: rtl; text-align: right; font-family: 'Vazirmatn', sans-serif;}
.stApp {background-color:#0d1117; color:#e0e0e0;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>نمودار استراتژی آپشن</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align:center; color:#888;'>تمامی حقوق مادی و معنوی این صفحه برای علیرضا حق وردی می باشد</h3>",
    unsafe_allow_html=True
)

# ----------------------------------
# Inputs
# ----------------------------------
c1,c2,c3,c4 = st.columns(4)
S = c1.number_input("قیمت دارایی پایه", value=1000.0)
iv = c2.number_input("IV (%)", value=30.0)
days_left = c3.number_input("روز تا سررسید", value=45)
r = c4.number_input("نرخ بدون ریسک (%)", value=4.0)/100

st.divider()

# ----------------------------------
# Leg Form
# ----------------------------------
edit_leg = st.session_state.legs[st.session_state.edit_idx] if st.session_state.edit_idx is not None else None
with st.expander("مدیریت Legها", expanded=True):
    with st.form("leg_form"):
        cols = st.columns(6)
        action = cols[0].selectbox("جهت", ["Buy","Sell"], index=0 if not edit_leg or edit_leg["Qty"]>0 else 1)
        opt_type = cols[1].selectbox("نوع", ["Call","Put"], index=0 if not edit_leg or edit_leg["Type"]=="Call" else 1)
        strike = cols[2].number_input("Strike", value=edit_leg["Strike"] if edit_leg else S)
        premium = cols[3].number_input("Premium", value=edit_leg["Premium"] if edit_leg else 20.0)
        qty = cols[4].number_input("تعداد", value=abs(edit_leg["Qty"]) if edit_leg else 1.0, min_value=0.01)
        margin = cols[5].number_input("Margin (Sell)", value=edit_leg["Margin"] if edit_leg else 0.0)

        if st.form_submit_button("ذخیره Leg"):
            leg = {
                "Action": action,
                "Type": opt_type,
                "Strike": strike,
                "Premium": premium,
                "Qty": qty if action=="Buy" else -qty,
                "Margin": margin if action=="Sell" else 0
            }
            if st.session_state.edit_idx is not None:
                st.session_state.legs[st.session_state.edit_idx] = leg
                st.session_state.edit_idx = None
            else:
                st.session_state.legs.append(leg)
            st.rerun()

# ----------------------------------
# Legs Table
# ----------------------------------
if not st.session_state.legs:
    st.info("حداقل یک Leg اضافه کنید.")
    st.stop()

st.subheader("📋 Legهای فعال")

rows = []
total_margin = 0
for i, l in enumerate(st.session_state.legs):
    side = "Buy" if l["Qty"]>0 else "Sell"
    rows.append({
        "جهت": side,
        "نوع": l["Type"],
        "Strike": l["Strike"],
        "Premium": l["Premium"],
        "Qty": l["Qty"],
        "Margin": l["Margin"],
        "idx": i
    })
    total_margin += l["Margin"]

df = pd.DataFrame(rows)

for _, row in df.iterrows():
    c = st.columns([5,1,1])
    color = "#00ff88" if row["Qty"]>0 else "#ff5555"
    c[0].markdown(
        f"<span style='color:{color}; font-weight:600;'>"
        f"{row['جهت']} {abs(row['Qty'])} {row['نوع']} @ {row['Strike']} | Premium {row['Premium']} | Margin {row['Margin']}"
        f"</span>",
        unsafe_allow_html=True
    )
    if c[1].button("✏️", key=f"edit_{row['idx']}"):
        st.session_state.edit_idx = int(row["idx"])
        st.rerun()
    if c[2].button("🗑", key=f"del_{row['idx']}"):
        del st.session_state.legs[int(row["idx"])]
        st.session_state.edit_idx = None
        st.rerun()

st.write(f"**مجموع Margin: {total_margin}**")

# ----------------------------------
# Payoff + Greeks
# ----------------------------------
prices = np.linspace(S*0.4, S*1.6, 800)
payoff = np.zeros_like(prices)
delta_curve = np.zeros_like(prices)
theta_curve = np.zeros_like(prices)

for leg in st.session_state.legs:
    payoff += option_payoff(prices, leg)
    for i,p in enumerate(prices):
        d,_,t,_ = bs_greeks(p, leg, iv, r, days_left)
        delta_curve[i] += d * leg["Qty"] * 100
        theta_curve[i] += t * leg["Qty"] * 100

profit = np.where(payoff>=0, payoff, np.nan)
loss = np.where(payoff<0, payoff, np.nan)

# Breakeven
idx = np.where(np.diff(np.sign(payoff)))[0]
breakevens = prices[idx]

# ----------------------------------
# Chart
# ----------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=prices, y=profit, name="سود",
    line=dict(color="#00ff88", width=3),
    hovertemplate="قیمت: %{x:.2f}<br>سود: %{y:.2f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=prices, y=loss, name="زیان",
    line=dict(color="#ff5555", width=3),
    hovertemplate="قیمت: %{x:.2f}<br>زیان: %{y:.2f}<extra></extra>"
))

# Breakeven lines
for be in breakevens:
    fig.add_vline(x=be, line=dict(color="#ffa500", dash="dot", width=2))
    fig.add_annotation(x=be, y=0, text=f"BE {be:.1f}", showarrow=True, arrowcolor="#ffa500")

fig.add_trace(go.Scatter(
    x=prices, y=delta_curve, name="Delta",
    yaxis="y2", visible="legendonly",
    line=dict(color="#00bfff", dash="dot"),
    hovertemplate="قیمت: %{x:.2f}<br>Delta: %{y:.2f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=prices, y=theta_curve, name="Theta",
    yaxis="y2", visible="legendonly",
    line=dict(color="#ffb000", dash="dash"),
    hovertemplate="قیمت: %{x:.2f}<br>Theta: %{y:.2f}<extra></extra>"
))

fig.add_hline(y=0, line_dash="dash", line_color="#888")

fig.add_annotation(
    text="ECON.CHAIN",
    x=0.5, y=0.5, xref="paper", yref="paper",
    showarrow=False, font=dict(size=48, color="#444"),
    opacity=0.18
)

fig.update_layout(
    hovermode="closest",
    xaxis_title="قیمت دارایی پایه",
    yaxis_title="سود / زیان",
    yaxis2=dict(title="Delta / Theta", overlaying="y", side="right"),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    height=750
)

st.plotly_chart(fig, use_container_width=True)
