"""
Equity Intrinsic Value Calculator — Two-Stage DCF Model
Compatible with Streamlit · Requires: streamlit, yfinance, plotly, pandas, numpy
Run: streamlit run equity_valuation_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Valuation | DCF Calculator",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — matches the reference app styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Step headers ── */
.step-header {
    font-size: 2rem; font-weight: 700; color: #1a1a2e;
    margin-top: 2rem; margin-bottom: 0.5rem;
}
/* ── Info callout ── */
.info-box {
    background: #e8f4f8; border-left: 4px solid #2196F3;
    padding: 0.9rem 1.1rem; border-radius: 6px;
    margin-bottom: 1rem; font-size: 0.93rem; color: #1a3a4a;
}
/* ── Formula code block ── */
.formula-block {
    background: #f7f7f8; border: 1px solid #ddd;
    border-radius: 8px; padding: 1.1rem 1.4rem;
    font-family: 'Courier New', monospace; font-size: 1rem;
    line-height: 2; color: #1a1a2e; margin-bottom: 1rem;
}
/* ── Result badge — green ── */
.result-green {
    background: #e8f5e9; border: 1px solid #4caf50;
    border-radius: 8px; padding: 0.7rem 1.1rem;
    font-size: 1.15rem; font-weight: 700; color: #2e7d32;
    margin-top: 0.6rem;
}
/* ── Result badge — red ── */
.result-red {
    background: #fce4ec; border: 1px solid #e91e63;
    border-radius: 8px; padding: 0.7rem 1.1rem;
    font-size: 1.15rem; font-weight: 700; color: #880e4f;
    margin-top: 0.6rem;
}
/* ── Metric pill (snapshot) ── */
.metric-label { font-size: 0.82rem; color: #555; font-weight: 500; }
.metric-value { font-size: 1.95rem; font-weight: 700; color: #1a1a2e; line-height: 1.1; }

/* ── Callout strip ── */
.callout-strip {
    background: #fff8e1; border-left: 4px solid #ffc107;
    padding: 0.75rem 1rem; border-radius: 6px;
    font-size: 0.95rem; margin: 0.6rem 0;
}
/* ── Section divider ── */
hr.section { border: none; border-top: 1px solid #e0e0e0; margin: 2rem 0; }

/* ── Overvalued / Undervalued ── */
.overvalued {
    background: #fce4ec; border-radius: 10px; padding: 1rem 1.5rem;
    border: 1.5px solid #e91e63;
}
.undervalued {
    background: #e8f5e9; border-radius: 10px; padding: 1rem 1.5rem;
    border: 1.5px solid #4caf50;
}
/* ── Table styling ── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
.styled-table th {
    background: #1a1a2e; color: white; padding: 0.55rem 0.8rem;
    text-align: right; font-weight: 600;
}
.styled-table th:first-child { text-align: center; }
.styled-table td {
    padding: 0.5rem 0.8rem; text-align: right;
    border-bottom: 1px solid #f0f0f0;
}
.styled-table td:first-child { text-align: center; font-weight: 600; }
.styled-table tr:nth-child(even) { background: #fafafa; }

/* ── Sensitivity table ── */
.sens-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.sens-table th {
    background: #1a1a2e; color: white; padding: 0.45rem 0.6rem;
    text-align: center; font-weight: 600;
}
.sens-table td { padding: 0.4rem 0.6rem; text-align: center; border: 1px solid #e0e0e0; }
.above-mkt  { background: #c8e6c9; color: #1b5e20; font-weight: 600; }
.below-mkt  { background: #ffcdd2; color: #b71c1c; font-weight: 600; }
.within-15  { background: #fff9c4; color: #f57f17; font-weight: 600; }
.base-case  { outline: 3px solid #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def fmt_large(val):
    """Format large numbers as $xT / $xB / $xM."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs_val >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


def safe_get(info, key, default=None):
    v = info.get(key, default)
    return default if v is None else v


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info
    hist  = t.history(period="2y")
    cf    = t.cashflow          # quarterly or annual; get annual
    inc   = t.income_stmt
    bs    = t.balance_sheet
    return info, hist, cf, inc, bs


def build_annual_financials(cf, inc, bs):
    """Return a tidy dict of annual financial series."""
    rows = {}
    # ── Cash Flow ──
    def _row(df, *keys):
        for k in keys:
            if df is not None and k in df.index:
                return df.loc[k]
        return None

    op_cf = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = _row(cf, "Capital Expenditure", "Capital Expenditures", "Purchase Of Plant And Equipment")
    rev   = _row(inc, "Total Revenue", "Revenue")
    ni    = _row(inc, "Net Income")
    tot_debt  = _row(bs, "Total Debt", "Long Term Debt")
    cash_eq   = _row(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")

    years = []
    if op_cf is not None:
        years = sorted([c for c in op_cf.index if hasattr(c, 'year')], key=lambda x: x.year)

    def _series(s):
        if s is None:
            return {}
        return {c.year: (s[c] / 1e6 if pd.notna(s[c]) else None) for c in s.index if hasattr(c, 'year')}

    op_cf_d = _series(op_cf)
    capex_d = _series(capex)
    rev_d   = _series(rev)
    ni_d    = _series(ni)

    # FCF = OpCF - abs(CapEx)
    all_years = sorted(set(op_cf_d) | set(capex_d))
    fcf_d = {}
    for yr in all_years:
        o = op_cf_d.get(yr)
        c = capex_d.get(yr)
        if o is not None and c is not None:
            fcf_d[yr] = o - abs(c)   # capex usually negative in yfinance

    # Net Debt (latest)
    latest_debt = None
    latest_cash = None
    if tot_debt is not None:
        for c in sorted(tot_debt.index, reverse=True):
            v = tot_debt[c]
            if pd.notna(v):
                latest_debt = v / 1e6
                break
    if cash_eq is not None:
        for c in sorted(cash_eq.index, reverse=True):
            v = cash_eq[c]
            if pd.notna(v):
                latest_cash = v / 1e6
                break

    net_debt = None
    if latest_debt is not None and latest_cash is not None:
        net_debt = latest_debt - latest_cash

    return {
        "op_cf": op_cf_d,
        "capex": {yr: abs(v) for yr, v in capex_d.items() if v is not None},
        "fcf": fcf_d,
        "revenue": rev_d,
        "net_income": ni_d,
        "net_debt": net_debt,
    }


def compute_fcf_growth(fcf_d):
    """Return YoY FCF growth rates."""
    years = sorted(fcf_d.keys())
    rates = {}
    for i in range(1, len(years)):
        y0, y1 = years[i-1], years[i]
        f0, f1 = fcf_d.get(y0), fcf_d.get(y1)
        if f0 and f1 and f0 != 0:
            rates[f"{y0}→{y1}"] = (f1 - f0) / abs(f0)
    return rates


def gauge_chart(value_pct: float, title: str = "WACC"):
    """Semi-circular gauge from 0–20%."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct * 100,
        number={"suffix": "%", "font": {"size": 38, "color": "#1a1a2e"}},
        gauge={
            "axis": {"range": [0, 20], "tickwidth": 1, "tickcolor": "#aaa",
                     "tickvals": [0, 5, 10, 15, 20]},
            "bar": {"color": "#1a1a2e", "thickness": 0.25},
            "steps": [
                {"range": [0, 5],  "color": "#c8e6c9"},
                {"range": [5, 10], "color": "#bbdefb"},
                {"range": [10, 15],"color": "#fff9c4"},
                {"range": [15, 20],"color": "#ffcdd2"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "value": value_pct * 100},
        },
        title={"text": title, "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def waterfall_chart(labels, values, title="Valuation Build-Up ($B)"):
    measures = ["absolute", "relative", "relative", "total"]
    vals_b = [v / 1000 for v in values]   # $M → $B
    text   = [f"${v/1000:.1f}B" for v in values]

    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        measure=measures,
        x=labels,
        y=vals_b,
        text=text,
        textposition="outside",
        connector={"line": {"color": "#ccc"}},
        increasing={"marker": {"color": "#43a047"}},
        decreasing={"marker": {"color": "#e53935"}},
        totals={"marker": {"color": "#43a047"}},
    ))
    fig.update_layout(
        title=title, showlegend=False, height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="$B",
    )
    return fig


def sensitivity_html(wacc_vals, g_vals, base_wacc, base_g, base_fcf_m,
                     n_yrs, near_g, mid_g, net_debt_m, shares_b,
                     near_yrs=3, market_price=None):
    """Build HTML sensitivity table."""
    def iv(w, g):
        if g >= w:
            return None
        fcf = base_fcf_m
        pv1 = 0
        for yr in range(1, n_yrs + 1):
            gr = near_g if yr <= near_yrs else mid_g
            fcf = fcf * (1 + gr)
            pv1 += fcf / (1 + w) ** yr
        tv  = fcf * (1 + g) / (w - g)
        pv_tv = tv / (1 + w) ** n_yrs
        eq_val = pv1 + pv_tv - net_debt_m
        return eq_val / (shares_b * 1000)

    mp = market_price or 0
    html = '<table class="sens-table"><tr><th>WACC \\ g →</th>'
    for g in g_vals:
        html += f"<th>{g*100:.1f}%</th>"
    html += "</tr>"

    for w in wacc_vals:
        html += f'<tr><td><b>{w*100:.1f}%</b></td>'
        for g in g_vals:
            val = iv(w, g)
            is_base = abs(w - base_wacc) < 0.001 and abs(g - base_g) < 0.001
            if val is None:
                cls = ""
                txt = "N/A"
            else:
                if mp > 0:
                    if val > mp * 1.15:
                        cls = "above-mkt"
                    elif val < mp * 0.85:
                        cls = "below-mkt"
                    else:
                        cls = "within-15"
                else:
                    cls = ""
                txt = f"${val:,.2f}"
            extra = ' style="outline:3px solid #1a1a2e;"' if is_base else ""
            html += f'<td class="{cls}"{extra}>{txt}</td>'
        html += "</tr>"
    html += "</table>"
    return html


# ──────────────────────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Equity Intrinsic Value Calculator")
st.markdown("**Two-Stage DCF Model** · FINA 4011/5011")
st.markdown('<div class="info-box">Enter a ticker below. The app fetches live financial data from Yahoo Finance, then walks you through every step of a two-stage DCF valuation.</div>', unsafe_allow_html=True)
st.markdown('<hr class="section">', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — SELECT A STOCK
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-header">Step 1 · Select a Stock</div>', unsafe_allow_html=True)

col_inp, col_btn = st.columns([3, 1])
with col_inp:
    ticker_input = st.text_input("Stock ticker (e.g. AAPL, MSFT, TSLA)", value="AAPL",
                                 label_visibility="visible")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    load_clicked = st.button("🔄 Load Data", use_container_width=True, type="primary")

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
    st.session_state.data_loaded = False

if load_clicked:
    st.session_state.ticker = ticker_input.strip().upper()
    st.session_state.data_loaded = True

if not st.session_state.data_loaded:
    st.info("👆 Enter a ticker and click **Load Data** to begin.")
    st.stop()

ticker_sym = st.session_state.ticker

with st.spinner(f"Fetching data for **{ticker_sym}**…"):
    try:
        info, hist, cf, inc, bs = load_ticker(ticker_sym)
    except Exception as e:
        st.error(f"Could not load data for **{ticker_sym}**: {e}")
        st.stop()

if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
    st.error(f"No data found for **{ticker_sym}**. Check the ticker and try again.")
    st.stop()

fin = build_annual_financials(cf, inc, bs)

# ── Derived values ──────────────────────────────────────────────────────────
current_price   = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice", 0)
market_cap      = safe_get(info, "marketCap", 0)
beta            = safe_get(info, "beta", 1.0) or 1.0
trailing_pe     = safe_get(info, "trailingPE")
forward_pe      = safe_get(info, "forwardPE")
ev_ebitda       = safe_get(info, "enterpriseToEbitda")
div_yield       = safe_get(info, "dividendYield", 0) or 0
shares_out      = safe_get(info, "sharesOutstanding", 0)  # raw shares
name            = safe_get(info, "longName", ticker_sym)
sector          = safe_get(info, "sector", "")
industry        = safe_get(info, "industry", "")
description     = safe_get(info, "longBusinessSummary", "")
analyst_target  = safe_get(info, "targetMeanPrice")
revenue_growth  = safe_get(info, "revenueGrowth")
earnings_growth = safe_get(info, "earningsGrowth")

shares_b = (shares_out or 0) / 1e9   # billions

# Base FCF (most recent year)
fcf_years = sorted(fin["fcf"].keys())
base_fcf_m = fin["fcf"].get(fcf_years[-1], 0) if fcf_years else 0   # $M
net_debt_m = fin["net_debt"] if fin["net_debt"] is not None else 0    # $M

# FCF historical growth rates
fcf_growth_rates = compute_fcf_growth(fin["fcf"])

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — COMPANY SNAPSHOT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.markdown('<div class="step-header">Step 2 · Company Snapshot</div>', unsafe_allow_html=True)

# Company name + sector
emoji_map = {
    "Technology": "🖥️", "Consumer Cyclical": "🛍️", "Healthcare": "⚕️",
    "Financial Services": "🏦", "Energy": "⚡", "Industrials": "🏭",
    "Communication Services": "📡", "Consumer Defensive": "🛒",
    "Basic Materials": "⛏️", "Real Estate": "🏢", "Utilities": "💡",
}
em = emoji_map.get(sector, "🏢")
st.markdown(f"## {em} {name} ({ticker_sym})")
if sector or industry:
    st.caption(f"{sector} · {industry}")

# Metric cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-label">Current Price</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${current_price:,.2f}</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-label">Market Cap</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(market_cap)}</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-label">Beta</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{beta:.2f}</div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-label">Trailing P/E</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{f"{trailing_pe:.1f}x" if trailing_pe else "N/A"}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

m5, m6, m7, m8 = st.columns(4)
with m5:
    st.markdown('<div class="metric-label">Forward P/E</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{f"{forward_pe:.1f}x" if forward_pe else "N/A"}</div>', unsafe_allow_html=True)
with m6:
    st.markdown('<div class="metric-label">EV/EBITDA</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A"}</div>', unsafe_allow_html=True)
with m7:
    st.markdown('<div class="metric-label">Dividend Yield</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{div_yield*100:.2f}%</div>', unsafe_allow_html=True)
with m8:
    st.markdown('<div class="metric-label">Shares Out</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{shares_b:.2f}B</div>', unsafe_allow_html=True)

# Business description (expandable)
if description:
    with st.expander("📋 Business Description"):
        st.write(description)

# 2-year price chart
if not hist.empty:
    st.markdown(f"**{ticker_sym} – 2-Year Price History**")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name="Close",
        line=dict(color="#1565C0", width=2),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
    ))
    fig_price.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date", yaxis_title="USD",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        showlegend=False,
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — FINANCIAL DATA
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.markdown('<div class="step-header">Step 3 · Financial Data (Auto-Retrieved)</div>', unsafe_allow_html=True)

cf_years = sorted(set(fin["op_cf"]) | set(fin["capex"]) | set(fin["fcf"]))
rev_years = sorted(set(fin["revenue"]) | set(fin["net_income"]))

col_cf, col_inc = st.columns(2)

with col_cf:
    st.markdown("**Cash Flow ($M)**")
    if cf_years:
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=[str(y) for y in cf_years],
            y=[fin["op_cf"].get(y, 0) for y in cf_years],
            name="Operating CF", marker_color="#1565C0",
        ))
        fig_cf.add_trace(go.Bar(
            x=[str(y) for y in cf_years],
            y=[fin["capex"].get(y, 0) for y in cf_years],
            name="CapEx", marker_color="#ef6c00",
        ))
        fig_cf.add_trace(go.Scatter(
            x=[str(y) for y in cf_years],
            y=[fin["fcf"].get(y) for y in cf_years],
            mode="lines+markers", name="FCF",
            line=dict(color="#2e7d32", width=2, dash="dot"),
            marker=dict(symbol="circle", size=8, color="#2e7d32"),
        ))
        fig_cf.update_layout(
            height=320, margin=dict(l=5,r=5,t=10,b=5),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_cf, use_container_width=True)

with col_inc:
    st.markdown("**Income ($M)**")
    if rev_years:
        fig_inc = go.Figure()
        fig_inc.add_trace(go.Bar(
            x=[str(y) for y in rev_years],
            y=[fin["revenue"].get(y, 0) for y in rev_years],
            name="Revenue", marker_color="#42a5f5",
        ))
        fig_inc.add_trace(go.Bar(
            x=[str(y) for y in rev_years],
            y=[fin["net_income"].get(y, 0) for y in rev_years],
            name="Net Income", marker_color="#66bb6a",
        ))
        fig_inc.update_layout(
            height=320, margin=dict(l=5,r=5,t=10,b=5),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_inc, use_container_width=True)

# Key callout
bcf_str  = fmt_large(base_fcf_m * 1e6) if base_fcf_m else "N/A"
nd_str   = fmt_large(net_debt_m * 1e6) if net_debt_m else "N/A"
st.markdown(
    f'<div class="callout-strip">📌 <b>Base FCF (most recent year):</b> {bcf_str} &nbsp;|&nbsp; '
    f'<b>Net Debt:</b> {nd_str}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-box"><b>FCF = Operating Cash Flow – Capital Expenditures.</b> '
    "This is cash available after maintaining/growing assets — the foundation of DCF.</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — WACC
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.markdown('<div class="step-header">Step 4 · Cost of Capital (WACC)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">WACC is the blended required return of all capital providers. '
    "It is the discount rate applied to future cash flows — a higher WACC lowers the valuation.</div>",
    unsafe_allow_html=True,
)

with st.expander("📐 Formulas"):
    st.markdown(
        '<div class="formula-block">'
        "WACC = (E/V) × Re + (D/V) × Rd × (1 − Tax Rate)<br><br>"
        "Cost of Equity (CAPM): Re = Rf + β × ERP"
        "</div>",
        unsafe_allow_html=True,
    )

col_capm, col_cap = st.columns(2)

with col_capm:
    st.markdown("### CAPM")
    rf = st.slider(
        "Risk-Free Rate Rf (%) — 10-yr Treasury, ~4.3% in 2025",
        min_value=0.0, max_value=10.0, value=4.30, step=0.05, format="%.2f",
    ) / 100
    erp = st.slider(
        "Equity Risk Premium ERP (%) — historical US avg ~5.5%",
        min_value=0.0, max_value=15.0, value=5.50, step=0.05, format="%.2f",
    ) / 100
    st.markdown(f"**Beta β (live: {beta:.2f})**")
    user_beta = st.number_input("", value=float(round(beta, 2)), step=0.01,
                                 min_value=0.0, max_value=10.0, label_visibility="collapsed")

    re = rf + user_beta * erp
    st.markdown(
        f'<div class="result-green">Cost of Equity: {re*100:.2f}%</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"= {rf*100:.1f}% + {user_beta:.2f} × {erp*100:.1f}%")

with col_cap:
    st.markdown("### Capital Structure")
    ev_pct = st.slider(
        "Equity Weight E/V (%)", min_value=0, max_value=100, value=80, step=1,
    )
    dv_pct = 100 - ev_pct
    st.markdown(f"⇒ **Debt Weight D/V = {dv_pct}%**")
    rd = st.slider(
        "Pre-Tax Cost of Debt Rd (%)", min_value=0.0, max_value=20.0, value=4.00, step=0.25, format="%.2f",
    ) / 100
    tax_rate = st.slider(
        "Effective Tax Rate (%)", min_value=0, max_value=50, value=21, step=1,
    ) / 100

    wacc = (ev_pct / 100) * re + (dv_pct / 100) * rd * (1 - tax_rate)
    st.markdown(
        f'<div class="result-green">WACC: {wacc*100:.2f}%</div>',
        unsafe_allow_html=True,
    )

st.plotly_chart(gauge_chart(wacc, f"WACC = {wacc*100:.2f}%"), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — FCF GROWTH ASSUMPTIONS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.markdown('<div class="step-header">Step 5 · FCF Growth Assumptions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box"><b>Two-stage:</b> Stage 1 = explicit year-by-year growth you set. '
    "Stage 2 = terminal value via Gordon Growth Model. The sliders are pre-populated with "
    "<b>actual historical growth rates</b> for this company — you can adjust them freely.</div>",
    unsafe_allow_html=True,
)

# Historical growth data (expandable)
with st.expander(f"📊 {ticker_sym} Actual Growth Data — expand to see before setting assumptions"):
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.markdown("**📈 Historical FCF Growth**")
        if fcf_growth_rates:
            for period, rate in sorted(fcf_growth_rates.items(), reverse=True):
                color = "green" if rate > 0 else "red"
                st.markdown(f":{color}[**{period}: {rate*100:+.1f}%**]")
        else:
            st.write("Insufficient FCF history.")

    with g_col2:
        st.markdown("**🔭 Analyst & Forward Estimates**")
        if earnings_growth is not None:
            st.markdown(f"Earnings Growth (TTM): &nbsp; **{earnings_growth*100:.1f}%**")
        if revenue_growth is not None:
            st.markdown(f"Revenue Growth (TTM): &nbsp; **{revenue_growth*100:.1f}%**")
        if analyst_target:
            st.markdown(f"Analyst Price Target: &nbsp; **${analyst_target:,.2f}**")

    with g_col3:
        st.markdown("**📦 FCF History ($M)**")
        if fin["fcf"]:
            fig_fcf_hist = go.Figure(go.Bar(
                x=[str(y) for y in sorted(fin["fcf"])],
                y=[fin["fcf"][y] for y in sorted(fin["fcf"])],
                marker_color="#43a047",
                text=[f"${v:,.0f}" for v in [fin["fcf"][y] for y in sorted(fin["fcf"])]],
                textposition="outside",
            ))
            fig_fcf_hist.update_layout(height=200, margin=dict(l=5,r=5,t=10,b=5),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        showlegend=False, yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
            st.plotly_chart(fig_fcf_hist, use_container_width=True)

# Stage 1 & 2 inputs
col_s1, col_s2 = st.columns(2)

# Suggest default growth: cap the most recent FCF growth at a reasonable level
recent_rates = list(fcf_growth_rates.values())
default_near = min(max(recent_rates[-1] * 100 if recent_rates else 20, -50), 100) if recent_rates else 20
default_mid  = max(default_near * 0.6, 5)

with col_s1:
    st.markdown("### Stage 1 — Explicit Period")
    n_yrs = st.slider("Forecast years", min_value=3, max_value=10, value=5, step=1)
    custom_toggle = st.toggle("Custom rate per year (override per-year)")

    if custom_toggle:
        custom_rates = []
        for yr in range(1, n_yrs + 1):
            r_yr = st.number_input(
                f"Year {yr} growth rate (%)", min_value=-50.0, max_value=200.0,
                value=round(default_near, 1), step=0.5, key=f"cr_{yr}",
            ) / 100
            custom_rates.append(r_yr)
        near_g = None
        mid_g  = None
        near_yrs = 0
    else:
        custom_rates = None
        near_yrs_label = min(3, n_yrs)
        near_g = st.slider(
            f"Near-term growth Yrs 1–{near_yrs_label} (%) — {ticker_sym} hist avg: {default_near:.1f}%",
            min_value=-50.0, max_value=200.0,
            value=round(min(max(default_near, -50), 100), 1), step=0.5,
        ) / 100
        near_yrs = near_yrs_label
        if n_yrs > near_yrs_label:
            mid_g = st.slider(
                f"Mid-term growth Yrs {near_yrs_label+1}+ (%) — suggested: {default_mid:.1f}%",
                min_value=-50.0, max_value=200.0,
                value=round(min(max(default_mid, -50), 100), 1), step=0.5,
            ) / 100
        else:
            mid_g = near_g

with col_s2:
    st.markdown("### Stage 2 — Terminal")
    perp_g = st.slider(
        "Perpetuity Growth Rate (%)", min_value=0.0, max_value=8.0, value=2.50, step=0.25,
    ) / 100
    st.markdown(
        '<div class="formula-block" style="font-size:0.95rem; padding:0.7rem 1rem;">'
        "TV = FCF_n × (1+g) / (WACC − g)"
        "</div>",
        unsafe_allow_html=True,
    )

# Base FCF (editable)
st.markdown("### Base FCF")
base_fcf_override = st.number_input(
    f"Base FCF ($M) — auto-pulled: {base_fcf_m:,.1f}M",
    value=float(round(base_fcf_m, 2)),
    step=100.0,
    format="%.2f",
)
base_fcf_m = base_fcf_override

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — DCF RESULTS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.markdown('<div class="step-header">Step 6 · DCF Results</div>', unsafe_allow_html=True)
st.markdown("### 📋 Year-by-Year Projections")

# Build projection table
rows = []
fcf_val = base_fcf_m
for yr in range(1, n_yrs + 1):
    if custom_rates:
        gr = custom_rates[yr - 1]
    else:
        gr = near_g if yr <= near_yrs else mid_g
    fcf_val = fcf_val * (1 + gr)
    df_factor = 1 / (1 + wacc) ** yr
    pv = fcf_val * df_factor
    rows.append({
        "Year": yr,
        "Growth Rate": f"{gr*100:.1f}%",
        "FCF ($M)": fcf_val,
        "Discount Factor": df_factor,
        "PV of FCF ($M)": pv,
    })

proj_df = pd.DataFrame(rows)

# Render as styled HTML table
html_table = '<table class="styled-table"><tr>'
headers = ["Year", "Growth Rate", "FCF ($M)", "Discount Factor", "PV of FCF ($M)"]
for h in headers:
    html_table += f"<th>{h}</th>"
html_table += "</tr>"
for _, row in proj_df.iterrows():
    html_table += "<tr>"
    html_table += f'<td>{int(row["Year"])}</td>'
    html_table += f'<td>{row["Growth Rate"]}</td>'
    html_table += f'<td>{row["FCF ($M)"]:,.1f}</td>'
    html_table += f'<td>{row["Discount Factor"]:.4f}</td>'
    html_table += f'<td>{row["PV of FCF ($M)"]:,.1f}</td>'
    html_table += "</tr>"
html_table += "</table>"
st.markdown(html_table, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Aggregates
pv_stage1 = proj_df["PV of FCF ($M)"].sum()
last_fcf  = proj_df["FCF ($M)"].iloc[-1]
tv_gross  = last_fcf * (1 + perp_g) / (wacc - perp_g) if wacc > perp_g else 0
pv_tv     = tv_gross / (1 + wacc) ** n_yrs
ev        = pv_stage1 + pv_tv
equity_val = ev - net_debt_m
iv_per_share = equity_val / (shares_b * 1000) if shares_b > 0 else 0

# Summary metrics
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown('<div class="metric-label">PV Stage 1 FCFs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(pv_stage1*1e6)}</div>', unsafe_allow_html=True)
with r2:
    st.markdown('<div class="metric-label">Terminal Value (gross)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(tv_gross*1e6)}</div>', unsafe_allow_html=True)
with r3:
    st.markdown('<div class="metric-label">PV of Terminal Value</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(pv_tv*1e6)}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

r4, r5, r6 = st.columns(3)
with r4:
    st.markdown('<div class="metric-label">Enterprise Value</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(ev*1e6)}</div>', unsafe_allow_html=True)
with r5:
    st.markdown('<div class="metric-label">(–) Net Debt</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(net_debt_m*1e6)}</div>', unsafe_allow_html=True)
with r6:
    st.markdown('<div class="metric-label">Equity Value</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{fmt_large(equity_val*1e6)}</div>', unsafe_allow_html=True)

st.markdown('<hr class="section">', unsafe_allow_html=True)

# Intrinsic Value Per Share
col_iv, col_chart = st.columns([1, 1.4])

with col_iv:
    st.markdown("### 🎯 Intrinsic Value Per Share")
    st.markdown(
        f'<div style="font-size:2.8rem; font-weight:700; color:#1a1a2e;">${iv_per_share:,.2f}</div>',
        unsafe_allow_html=True,
    )
    premium = (current_price - iv_per_share) / iv_per_share if iv_per_share != 0 else 0
    pct_str = f"{abs(premium)*100:.1f}%"
    arrow = "🔻" if premium > 0 else "🔼"
    dir_str = "premium" if premium > 0 else "discount"
    st.markdown(
        f'<div style="font-size:1rem; color:{"#e53935" if premium>0 else "#43a047"}; font-weight:600;">'
        f'{arrow} {pct_str} {dir_str} vs market ${current_price:,.2f}</div>',
        unsafe_allow_html=True,
    )

    if premium > 0.05:
        st.markdown(
            f'<div class="overvalued">❌ <b>POTENTIALLY OVERVALUED</b><br>'
            f'Trading at a {pct_str} premium to intrinsic value.</div>',
            unsafe_allow_html=True,
        )
    elif premium < -0.05:
        st.markdown(
            f'<div class="undervalued">✅ <b>POTENTIALLY UNDERVALUED</b><br>'
            f'Trading at a {pct_str} discount to intrinsic value.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="callout-strip">🟡 <b>FAIRLY VALUED</b><br>'
            f'Trading within 5% of intrinsic value.</div>',
            unsafe_allow_html=True,
        )

with col_chart:
    fig_wf = waterfall_chart(
        labels=["PV Stage 1", "PV Terminal", "Enterprise Value", "Less Net Debt", "Equity Value"],
        values=[pv_stage1, pv_tv, ev, -net_debt_m, equity_val],
        title="Valuation Build-Up ($B)",
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — SENSITIVITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.markdown('<div class="step-header">Step 7 · Sensitivity Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">Intrinsic value per share across a range of WACC and terminal growth inputs. '
    "🟢 Above market price · 🔴 Below market price · 🟡 Within 15%</div>",
    unsafe_allow_html=True,
)

wacc_range = [wacc - 0.04, wacc - 0.02, wacc, wacc + 0.02, wacc + 0.04, wacc + 0.06]
wacc_range = [max(w, 0.01) for w in wacc_range]
g_range    = [perp_g - 0.015, perp_g - 0.01, perp_g - 0.005, perp_g,
              perp_g + 0.005, perp_g + 0.01, perp_g + 0.015]
g_range    = [max(g, 0.005) for g in g_range]

eff_near_g = near_g if not custom_rates else sum(custom_rates[:near_yrs]) / max(near_yrs, 1)
eff_mid_g  = mid_g if not custom_rates else sum(custom_rates[near_yrs:]) / max(n_yrs - near_yrs, 1)

sens_html = sensitivity_html(
    wacc_vals=wacc_range,
    g_vals=g_range,
    base_wacc=wacc,
    base_g=perp_g,
    base_fcf_m=base_fcf_m,
    n_yrs=n_yrs,
    near_g=eff_near_g,
    mid_g=eff_mid_g,
    net_debt_m=net_debt_m,
    shares_b=shares_b,
    near_yrs=near_yrs,
    market_price=current_price,
)
st.markdown(sens_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Base case (current assumptions) outlined in bold. Values computed using same Stage 1/2 assumptions.")

# Footer
st.markdown('<hr class="section">', unsafe_allow_html=True)
st.caption("Data sourced from Yahoo Finance via yfinance. For educational purposes only — not investment advice.")
