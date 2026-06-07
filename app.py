import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model dari .pkl (identik dengan notebook) ─────────────────────────────
@st.cache_resource
def load_model():
    pkl_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
    return joblib.load(pkl_path)

model = load_model()

# ── Rekonstruksi feature names persis seperti di notebook ─────────────────────
@st.cache_resource
def get_feature_names():
    pre = model.named_steps['preprocessor']
    numeric_features = ['Study_Hours_per_Day','Attendance_Rate','Assignment_Delay_Days',
                        'Travel_Time_Minutes','Stress_Index','GPA']
    cat_features     = ['Internet_Access','Part_Time_Job','Scholarship','Semester']
    ohe = pre.named_transformers_['cat'].named_steps['onehot']
    encoded = ohe.get_feature_names_out(cat_features).tolist()
    return numeric_features + encoded

feature_names = get_feature_names()

# ── Background data untuk SHAP LinearExplainer ────────────────────────────────
@st.cache_resource
def get_explainer():
    import shap
    pre = model.named_steps['preprocessor']
    lr  = model.named_steps['model']
    np.random.seed(42)
    n = 300
    bg = pd.DataFrame({
        'Study_Hours_per_Day':   np.random.uniform(0, 10, n),
        'Attendance_Rate':       np.random.uniform(40, 100, n),
        'Assignment_Delay_Days': np.random.uniform(0, 15, n),
        'Travel_Time_Minutes':   np.random.uniform(5, 120, n),
        'Stress_Index':          np.random.uniform(1, 10, n),
        'GPA':                   np.random.uniform(1.5, 4.0, n),
        'Internet_Access':       np.random.choice(['Yes','No'], n),
        'Part_Time_Job':         np.random.choice(['Yes','No'], n),
        'Scholarship':           np.random.choice(['Yes','No'], n),
        'Semester':              np.random.choice(['Year 1','Year 2','Year 3','Year 4'], n),
    })
    bg_t = pre.transform(bg)
    return shap.LinearExplainer(lr, bg_t, feature_perturbation="interventional")

explainer = get_explainer()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --paper:    #FDFCF8;
    --ink:      #1A1A18;
    --muted:    #6B6B66;
    --line:     #E6E2D9;
    --accent:   #7A6E5A;
    --olive:    #71713B;
    --steel:    #94A5B3;
    --bone:     #E2DCD0;
    --risk:     #8C3A25;
    --risk-bg:  #F8F0EC;
    --risk-ln:  #C9A090;
    --safe:     #2E5C42;
    --safe-bg:  #EBF2ED;
    --safe-ln:  #90BAA2;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--paper);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}

/* ── hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main > div.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }

/* ── HEADER ── */
.hdr {
    padding: 3.5rem clamp(1.8rem, 10vw, 8rem) 3rem;
    border-bottom: 1px solid var(--line);
    background: var(--paper);
}
.hdr-eyebrow {
    font-size: 0.62rem;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.hdr-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 5.5vw, 4.2rem);
    font-weight: 400;
    font-style: italic;
    line-height: 1.06;
    color: var(--ink);
    margin-bottom: 0.9rem;
}
.hdr-sub {
    font-size: 0.82rem;
    line-height: 1.72;
    color: var(--muted);
    font-weight: 300;
    max-width: 480px;
}

/* ── PAGE BODY ── */
.pg {
    padding: 2.8rem clamp(1.8rem, 10vw, 8rem) 6rem;
}

/* ── SECTION LABEL ── */
.sl {
    font-size: 0.6rem;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.8rem;
}

/* ── WIDGET LABELS ── */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.66rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child,
[data-testid="stSlider"] div[role="progressbar"] {
    background: var(--line) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background: var(--olive) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--olive) !important;
    border: 3px solid var(--paper) !important;
    box-shadow: 0 0 0 1.5px var(--olive) !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
}
[data-testid="stSlider"] p {
    font-size: 0.64rem !important;
    color: var(--muted) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── SELECTBOX ── */
[data-baseweb="select"] > div {
    border-color: var(--line) !important;
    border-radius: 2px !important;
    background: var(--paper) !important;
    font-size: 0.84rem !important;
}

/* ── RADIO ── */
[data-baseweb="radio"] [data-state="checked"] div,
[data-baseweb="radio"] div[class*="radioInner"] {
    background: var(--olive) !important;
}
[data-baseweb="radio"] label span:first-child {
    border-color: var(--olive) !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] label {
    font-size: 0.83rem !important;
    color: var(--ink) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.66rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    transition: background 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover { background: var(--accent) !important; }

/* ── RESULT BOX ── */
.rbox {
    padding: 2rem 2rem 1.8rem;
    border: 1px solid var(--line);
}
.rbox.risk { background: var(--risk-bg); border-left: 3px solid var(--risk); }
.rbox.safe { background: var(--safe-bg); border-left: 3px solid var(--safe); }
.r-eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.rbox.risk .r-eyebrow { color: var(--risk); }
.rbox.safe .r-eyebrow { color: var(--safe); }
.r-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem;
    font-weight: 300;
    font-style: italic;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.rbox.risk .r-pct { color: var(--risk); }
.rbox.safe .r-pct { color: var(--safe); }
.r-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-style: italic;
    font-weight: 400;
    margin-bottom: 0.6rem;
}
.rbox.risk .r-verdict { color: var(--risk); }
.rbox.safe .r-verdict { color: var(--safe); }
.r-note {
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.6;
    font-weight: 300;
}

/* ── GAUGE ── */
.gauge { margin-top: 1.8rem; }
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.gauge-track {
    height: 2px;
    background: var(--line);
    position: relative;
    overflow: visible;
}
.gauge-fill { height: 100%; }
.gauge-dot {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 8px; height: 8px;
    border-radius: 50%;
    border: 2px solid var(--paper);
    box-shadow: 0 0 0 1px currentColor;
}

/* ── SHAP ROWS ── */
.shap-table { width: 100%; }
.shap-row {
    display: grid;
    grid-template-columns: 170px 1fr 60px;
    align-items: center;
    gap: 1rem;
    padding: 0.58rem 0;
    border-bottom: 1px solid var(--line);
}
.shap-row:last-child { border-bottom: none; }
.shap-name {
    font-size: 0.74rem;
    color: var(--ink);
    font-weight: 400;
}
.shap-bar-wrap {
    position: relative;
    height: 4px;
    background: var(--line);
}
/* positive SHAP = pushes toward dropout = risk color (right of center) */
.shap-bar-pos {
    position: absolute;
    height: 100%;
    left: 50%;
}
/* negative SHAP = pushes toward safe = safe color (left of center) */
.shap-bar-neg {
    position: absolute;
    height: 100%;
    right: 50%;
}
.shap-center-line {
    position: absolute;
    left: 50%;
    top: -3px;
    width: 1px;
    height: 10px;
    background: var(--muted);
    opacity: 0.4;
}
.shap-val {
    font-size: 0.68rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
}

/* ── ABOUT ── */
.ablock {
    padding: 1.4rem 1.5rem;
    border: 1px solid var(--line);
    background: var(--paper);
}
.ablock h4 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-style: italic;
    font-weight: 400;
    color: var(--ink);
    padding-bottom: 0.6rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--line);
}
.ablock p, .ablock li {
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.78;
    font-weight: 300;
}
.ablock ul { padding-left: 1rem; }
.ablock li { margin-bottom: 0.1rem; }

/* ── FOOTER ── */
.ftr {
    border-top: 1px solid var(--line);
    padding: 1.4rem clamp(1.8rem, 10vw, 8rem);
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: var(--paper);
}

/* spacing */
.s1 { margin-bottom: 0.8rem; }
.s2 { margin-bottom: 1.6rem; }
.s3 { margin-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">Educational Analytics &mdash; Machine Learning</div>
  <div class="hdr-title">Student Dropout<br>Predictor</div>
  <div class="hdr-sub">
    Predict whether a student is at risk of dropping out using logistic regression
    trained on academic and lifestyle indicators. Results are explained using SHAP values.
  </div>
</div>
""", unsafe_allow_html=True)

# ── PAGE BODY ─────────────────────────────────────────────────────────────────
st.markdown('<div class="pg">', unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="sl">Student Profile</div>', unsafe_allow_html=True)

L, _g, R = st.columns([10, 1, 10])

with L:
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    gpa              = st.slider("GPA", 1.0, 4.0, 2.8, 0.05)
    attendance       = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)
    study_hours      = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, 0.25)
    assignment_delay = st.slider("Assignment Delay (Days)", 0, 15, 3)
    semester         = st.selectbox("Current Year", ["Year 1","Year 2","Year 3","Year 4"])

with R:
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    stress      = st.slider("Stress Index (1–10)", 1.0, 10.0, 5.0, 0.1)
    travel_time = st.slider("Travel Time to Campus (min)", 0, 120, 30)
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    internet    = st.radio("Internet Access",  ["Yes","No"], horizontal=True)
    part_time   = st.radio("Part-Time Job",    ["Yes","No"], horizontal=True)
    scholarship = st.radio("Scholarship",      ["Yes","No"], horizontal=True)

st.markdown('<div class="s2"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([6, 6, 6])
with btn_col:
    clicked = st.button("Run Prediction", use_container_width=True)

st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────────────────────────────
if clicked:
    import shap

    inp = pd.DataFrame({
        'Study_Hours_per_Day':    [study_hours],
        'Attendance_Rate':        [attendance],
        'Assignment_Delay_Days':  [float(assignment_delay)],
        'Travel_Time_Minutes':    [float(travel_time)],
        'Stress_Index':           [stress],
        'GPA':                    [gpa],
        'Internet_Access':        [internet],
        'Part_Time_Job':          [part_time],
        'Scholarship':            [scholarship],
        'Semester':               [semester],
    })

    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    dp    = proba[1]   # dropout probability

    # SHAP values for this prediction
    pre   = model.named_steps['preprocessor']
    inp_t = pre.transform(inp)
    sv    = explainer.shap_values(inp_t)   # shape (1, n_features)

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap':    sv[0],
    })
    shap_df['abs'] = shap_df['shap'].abs()
    shap_df = shap_df.sort_values('abs', ascending=False).head(10).reset_index(drop=True)

    # Max abs SHAP for normalizing bar widths (cap at 50% each side)
    max_abs = shap_df['abs'].max() if shap_df['abs'].max() > 0 else 1.0

    # ── Layout ──
    st.markdown('<div class="sl">Result</div>', unsafe_allow_html=True)
    col_res, col_shap = st.columns([5, 7])

    # ── Result box ──
    with col_res:
        if pred == 1:
            st.markdown(f"""
            <div class="rbox risk">
              <div class="r-eyebrow">Dropout Probability</div>
              <div class="r-pct">{dp:.1%}</div>
              <div class="r-verdict">At Risk of Dropout</div>
              <div class="r-note">This student shows a high likelihood of dropping out.<br>
              Early intervention is recommended.</div>
              <div class="gauge">
                <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{dp*100:.1f}%;background:var(--risk);"></div>
                  <div class="gauge-dot" style="left:{dp*100:.1f}%;color:var(--risk);"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rbox safe">
              <div class="r-eyebrow">Dropout Probability</div>
              <div class="r-pct">{dp:.1%}</div>
              <div class="r-verdict">Not at Risk</div>
              <div class="r-note">This student is unlikely to drop out based on<br>
              current academic and lifestyle indicators.</div>
              <div class="gauge">
                <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{dp*100:.1f}%;background:var(--safe);"></div>
                  <div class="gauge-dot" style="left:{dp*100:.1f}%;color:var(--safe);"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── SHAP contributing factors ──
    with col_shap:
        st.markdown('<div class="sl">Top 10 Contributing Factors (SHAP)</div>', unsafe_allow_html=True)

        # Legend
        st.markdown("""
        <div style="display:flex;gap:1.4rem;margin-bottom:1.2rem;font-size:0.64rem;
                    letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);">
          <span style="display:flex;align-items:center;gap:0.4rem;">
            <span style="display:inline-block;width:18px;height:3px;background:var(--risk);"></span>
            Increases risk
          </span>
          <span style="display:flex;align-items:center;gap:0.4rem;">
            <span style="display:inline-block;width:18px;height:3px;background:var(--safe);"></span>
            Decreases risk
          </span>
        </div>
        """, unsafe_allow_html=True)

        rows_html = '<div class="shap-table">'
        for _, row in shap_df.iterrows():
            raw_name = row['feature']
            sv_val   = row['shap']
            # Clean up OHE feature names for display
            display_name = raw_name.replace('_',' ').replace(' No','  = No').replace(' Yes','  = Yes')
            for yr in ['Year 1','Year 2','Year 3','Year 4']:
                display_name = display_name.replace(f'Semester {yr}', f'Semester = {yr}')

            bar_pct  = min(row['abs'] / max_abs * 46, 46)  # max 46% each side
            if sv_val >= 0:
                bar_html = f'<div class="shap-bar-pos" style="width:{bar_pct:.1f}%;background:var(--risk);"></div>'
                val_color = "var(--risk)"
                sign = "+"
            else:
                bar_html = f'<div class="shap-bar-neg" style="width:{bar_pct:.1f}%;background:var(--safe);"></div>'
                val_color = "var(--safe)"
                sign = ""

            rows_html += f"""
            <div class="shap-row">
              <span class="shap-name">{display_name}</span>
              <div class="shap-bar-wrap">
                {bar_html}
                <div class="shap-center-line"></div>
              </div>
              <span class="shap-val" style="color:{val_color};">{sign}{sv_val:.3f}</span>
            </div>"""

        rows_html += '</div>'
        st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:0.65rem;color:var(--muted);margin-top:1rem;font-weight:300;line-height:1.6;">
        SHAP values show each feature's contribution to the prediction.
        Positive values push toward dropout; negative values push toward safe.
        </p>
        """, unsafe_allow_html=True)

    st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

# ── ABOUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sl">About</div>', unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("""
    <div class="ablock">
      <h4>Algorithm</h4>
      <p>Logistic Regression trained with SMOTE oversampling to handle class imbalance.
      Selected for superior Recall — critical for catching at-risk students
      before they drop out.</p>
    </div>""", unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="ablock">
      <h4>Features Used</h4>
      <ul>
        <li>Study Hours per Day</li>
        <li>Attendance Rate</li>
        <li>Assignment Delay Days</li>
        <li>Travel Time to Campus</li>
        <li>Stress Index</li>
        <li>GPA</li>
        <li>Internet Access, Part-Time Job</li>
        <li>Scholarship, Semester</li>
      </ul>
    </div>""", unsafe_allow_html=True)

with a3:
    st.markdown("""
    <div class="ablock">
      <h4>Explainability (SHAP)</h4>
      <p>Each prediction is explained using SHAP (SHapley Additive exPlanations),
      which shows exactly how much each feature pushed the model toward
      or away from a dropout prediction for that specific student.</p>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
  Student Dropout Predictor &nbsp;&middot;&nbsp; Logistic Regression + SHAP &nbsp;&middot;&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
