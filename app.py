import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Memuat model dari file JSON
# ------------------------------------------------------------
@st.cache_resource
def load_model_from_json():
    with open("model_data.json", "r") as f:
        data = json.load(f)
    
    # Rebuild preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, data['numeric_features']),
            ('cat', categorical_transformer, data['categorical_features'])
        ]
    )
    
    # Set parameter preprocessor dari data yang disimpan
    # Numeric imputer median
    preprocessor.named_transformers_['num'].named_steps['imputer'].statistics_ = np.array(data['median_values'])
    # Scaler mean & scale
    preprocessor.named_transformers_['num'].named_steps['scaler'].mean_ = np.array(data['scaler_mean'])
    preprocessor.named_transformers_['num'].named_steps['scaler'].scale_ = np.array(data['scaler_scale'])
    # One-hot encoder categories
    preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_ = [np.array(cat) for cat in data['categories']]
    
    # Koefisien dan intercept
    coef = np.array(data['coef'])
    intercept = data['intercept']
    feature_names = data['feature_names']
    
    return preprocessor, coef, intercept, feature_names

preprocessor, coef, intercept, feature_names = load_model_from_json()

# Fungsi prediksi manual
def predict_proba(input_df):
    X = preprocessor.transform(input_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    linear = np.dot(X, coef) + intercept
    proba_dropout = 1 / (1 + np.exp(-linear))
    proba_safe = 1 - proba_dropout
    return np.column_stack((proba_safe, proba_dropout))

class DummyModel:
    def predict_proba(self, X):
        return predict_proba(X)
    def predict(self, X):
        return (predict_proba(X)[:,1] >= 0.5).astype(int)

model = DummyModel()

# ------------------------------------------------------------
# CSS – layout potrait
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600;1,700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --paper:     #FDFCF8;
    --ink:       #1A1A1A;
    --ink-light: #6B6B6B;
    --border:    #E8E5DC;
    --accent:    #8B7355;
    --risk:      #A0522D;
    --risk-bg:   #FBF5F0;
    --safe:      #4A6B5D;
    --safe-bg:   #F0F5F2;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--paper);
    color: var(--ink);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main > div.block-container {
    padding: 2rem 2rem !important;
    max-width: 700px !important;
    margin: 0 auto !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

.hdr {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hdr-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 0.75rem;
}
.hdr-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 500;
    font-style: italic;
    line-height: 1.1;
    color: var(--ink);
    margin-bottom: 0.5rem;
}
.hdr-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    line-height: 1.5;
    color: var(--ink-light);
    font-weight: 400;
    max-width: 85%;
}
.sl {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: var(--ink-light) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSlider"] {
    padding-bottom: 0.5rem !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--paper) !important;
    width: 16px !important;
    height: 16px !important;
}
[data-testid="stSlider"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    color: var(--ink-light) !important;
}
div[data-testid="stRadio"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 2rem !important;
    align-items: center !important;
    flex-wrap: wrap !important;
}
div[data-testid="stRadio"] > label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    margin: 0 !important;
    white-space: nowrap !important;
}
.row-widget.stRadio > div {
    gap: 2rem !important;
}
[data-baseweb="select"] > div {
    border-color: var(--border) !important;
    border-radius: 0px !important;
    background: var(--paper) !important;
}
[data-baseweb="select"]:hover > div {
    border-color: var(--accent) !important;
}
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 1.5rem !important;
    transition: background 0.2s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
}
.rbox {
    border-radius: 0px;
    padding: 1.5rem;
    background: var(--paper);
    border: 1px solid var(--border);
    margin-top: 1rem;
}
.rbox.risk { background: var(--risk-bg); border-left: 3px solid var(--risk); }
.rbox.safe { background: var(--safe-bg); border-left: 3px solid var(--safe); }
.r-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.rbox.risk .r-eyebrow { color: var(--risk); }
.rbox.safe .r-eyebrow { color: var(--safe); }
.r-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 600;
    font-style: italic;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.rbox.risk .r-pct { color: var(--risk); }
.rbox.safe .r-pct { color: var(--safe); }
.r-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 500;
    font-style: italic;
    margin-bottom: 0.5rem;
}
.gauge { margin-top: 1.2rem; }
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-light);
    margin-bottom: 0.5rem;
}
.gauge-track {
    height: 2px;
    background: var(--border);
    position: relative;
}
.gauge-fill { height: 100%; }
.gauge-dot {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid var(--paper);
    box-shadow: 0 0 0 1px currentColor;
}
.frow {
    display: grid;
    grid-template-columns: 140px 1fr 50px;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
}
.fn {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--ink);
    font-weight: 500;
}
.fb-bg {
    height: 2px;
    background: var(--border);
    overflow: hidden;
}
.fb-fill { height: 100%; }
.fp {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: var(--ink-light);
    text-align: right;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
.ftr {
    border-top: 1px solid var(--border);
    padding: 1.5rem 0;
    margin-top: 2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: var(--ink-light);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">Educational Analytics &mdash; Machine Learning</div>
  <div class="hdr-title">Student Dropout<br>Predictor</div>
  <div class="hdr-sub">
    Identify students at risk of dropping out using a logistic regression model
    trained on academic, personal and lifestyle indicators.
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.markdown('<div class="sl">Student Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gpa              = st.slider("GPA", 1.0, 4.0, 2.8, 0.05)
    attendance       = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)
    study_hours      = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, 0.25)
    assignment_delay = st.slider("Assignment Delay (Days)", 0, 15, 3)
    semester         = st.selectbox("Current Year", ["Year 1","Year 2","Year 3","Year 4"])

with col2:
    stress      = st.slider("Stress Index (1–10)", 1.0, 10.0, 5.0, 0.1)
    travel_time = st.slider("Travel Time to Campus (min)", 0, 120, 30)
    internet    = st.radio("Internet Access",  ["Yes","No"], horizontal=True)
    part_time   = st.radio("Part-Time Job",    ["Yes","No"], horizontal=True)
    scholarship = st.radio("Scholarship",      ["Yes","No"], horizontal=True)

st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    clicked = st.button("Run Prediction", use_container_width=True)

# ------------------------------------------------------------
# Prediksi dan tampilan hasil
# ------------------------------------------------------------
if clicked:
    inp = pd.DataFrame({
        'Internet_Access':        [internet],
        'Study_Hours_per_Day':    [study_hours],
        'Attendance_Rate':        [attendance],
        'Assignment_Delay_Days':  [float(assignment_delay)],
        'Travel_Time_Minutes':    [float(travel_time)],
        'Part_Time_Job':          [part_time],
        'Scholarship':            [scholarship],
        'Stress_Index':           [stress],
        'GPA':                    [gpa],
        'Semester':               [semester],
    })

    proba = model.predict_proba(inp)[0]
    dp = proba[1]

    # Hitung kontribusi fitur
    X_input = preprocessor.transform(inp)
    if hasattr(X_input, "toarray"):
        X_input = X_input.toarray()
    X_input = X_input.flatten()
    contributions = coef * X_input
    contrib_df = pd.DataFrame({
        'feature': feature_names,
        'contrib': contributions,
        'abs_contrib': np.abs(contributions),
    }).sort_values('abs_contrib', ascending=False)

    def clean_feature_name(name):
        if name.startswith('Semester_'):
            return "Semester: " + name.replace('Semester_', '')
        if name.startswith('Part_Time_Job_'):
            return "Part-Time Job: " + name.replace('Part_Time_Job_', '')
        if name.startswith('Internet_Access_'):
            return "Internet Access: " + name.replace('Internet_Access_', '')
        if name.startswith('Scholarship_'):
            return "Scholarship: " + name.replace('Scholarship_', '')
        return name

    contrib_df['feature_clean'] = contrib_df['feature'].apply(clean_feature_name)
    top10 = contrib_df.head(10).copy()
    total_abs = top10['abs_contrib'].sum()
    top10['pct'] = (top10['abs_contrib'] / total_abs) * 100

    st.markdown('<div class="sl">Result</div>', unsafe_allow_html=True)

    risk_class = "risk" if dp > 0.5 else "safe"
    verdict = "At Risk of Dropout" if dp > 0.5 else "Not at Risk"
    st.markdown(f"""
    <div class="rbox {risk_class}">
      <div class="r-eyebrow">Dropout Probability</div>
      <div class="r-pct">{dp:.1%}</div>
      <div class="r-verdict">{verdict}</div>
      <div class="gauge">
        <div class="gauge-labels"><span>Low risk</span><span>High risk</span></div>
        <div class="gauge-track">
          <div class="gauge-fill" style="width:{dp*100:.1f}%;background:{'#8c3a25' if dp>0.5 else '#2e5c42'};"></div>
          <div class="gauge-dot"  style="left:{dp*100:.1f}%;color:{'#8c3a25' if dp>0.5 else '#2e5c42'};"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sl" style="margin-top: 1.5rem;">Top 10 Contributing Factors</div>', unsafe_allow_html=True)

    rows = ""
    for _, row in top10.iterrows():
        contrib_val = row['contrib']
        pct = row['pct']
        clr = "#8c3a25" if contrib_val > 0 else "#2e5c42"
        sign = "+" if contrib_val > 0 else ""
        rows += f"""
        <div class="frow">
          <span class="fn">{row['feature_clean']}</span>
          <div class="fb-bg"><div class="fb-fill" style="width:{pct:.0f}%;background:{clr};"></div></div>
          <span class="fp">{sign}{pct:.0f}%</span>
        </div>
        """
    st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("""
<div class="ftr">
  Student Dropout Predictor &nbsp;&middot;&nbsp; Logistic Regression &nbsp;&middot;&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
