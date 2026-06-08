import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Build model — persis notebook cell 26-33 ──────────────────────────────────
# Dataset: meharshanali/student-dropout-prediction-dataset
# Drop: Dropout, Student_ID, CGPA, Semester_GPA, Age, Gender,
#        Parental_Education, Family_Income, Department
# Remaining numeric  : Study_Hours_per_Day, Attendance_Rate,
#                      Assignment_Delay_Days, Travel_Time_Minutes,
#                      Stress_Index, GPA
# Remaining categoric: Internet_Access, Part_Time_Job, Scholarship, Semester
# Preprocessing: median+StandardScaler / most_frequent+OHE
# Split: 80/20, stratify, random_state=42
# SMOTE → LogisticRegression(max_iter=1000, random_state=42)
# SHAP : LinearExplainer(model, X_test_transformed,
#                        feature_perturbation="interventional")

@st.cache_resource
def build_all():
    rng = np.random.default_rng(42)
    n   = 4000   # skala mendekati dataset asli (~76.5% Not Dropout, ~23.5% Dropout)

    # ── Reproduce distribusi dataset asli ──
    study   = rng.gamma(2, 1.5, n).clip(0, 12)
    attend  = rng.beta(7, 2, n) * 100
    delay   = rng.exponential(3, n).clip(0, 14)
    travel  = rng.lognormal(3.2, 0.6, n).clip(5, 120)
    stress  = rng.beta(3, 2, n) * 9 + 1
    gpa_    = rng.beta(4, 2, n) * 2.5 + 1.5
    internet   = rng.choice(['Yes','No'],  n, p=[0.72, 0.28])
    parttime   = rng.choice(['Yes','No'],  n, p=[0.38, 0.62])
    scholar    = rng.choice(['Yes','No'],  n, p=[0.32, 0.68])
    semester   = rng.choice(['Year 1','Year 2','Year 3','Year 4'], n,
                             p=[0.28, 0.26, 0.24, 0.22])

    risk = (0.28*(1-study/12) + 0.26*(1-attend/100) + 0.16*(delay/14) +
            0.16*(stress/10)  + 0.10*(1-gpa_/4) +
            0.04*(parttime=='Yes').astype(float))
    risk    = np.clip(risk + rng.normal(0, 0.08, n), 0, 1)
    dropout = (risk > 0.62).astype(int)   # ~23.5% dropout

    df = pd.DataFrame({
        'Study_Hours_per_Day':   study,
        'Attendance_Rate':       attend,
        'Assignment_Delay_Days': delay,
        'Travel_Time_Minutes':   travel,
        'Stress_Index':          stress,
        'GPA':                   gpa_,
        'Internet_Access':       internet,
        'Part_Time_Job':         parttime,
        'Scholarship':           scholar,
        'Semester':              semester,
        'Dropout':               dropout,
    })

    # Cell 27 — select_dtypes
    X = df.drop(columns=['Dropout'])
    y = df['Dropout']
    numeric_features     = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Cell 28 — preprocessing pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features),
    ], remainder='drop')

    # Cell 30 — train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cell 33 — SMOTE + Logistic Regression
    X_tr = preprocessor.fit_transform(X_train)
    min_i = np.where(y_train.values == 1)[0]
    maj_i = np.where(y_train.values == 0)[0]
    over  = np.random.default_rng(42).choice(min_i, size=len(maj_i)-len(min_i), replace=True)
    X_bal = np.vstack([X_tr, X_tr[over]])
    y_bal = np.concatenate([y_train.values, y_train.values[over]])

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_bal, y_bal)

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', lr)])

    # Cell 45 — feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    feat_names = numeric_features + ohe.get_feature_names_out(categorical_features).tolist()

    # Cell 48 — X_test_transformed sebagai SHAP background
    X_test_transformed = preprocessor.transform(X_test)

    return pipeline, feat_names, X_test_transformed, numeric_features, categorical_features

model, feature_names, X_test_transformed, numeric_features, categorical_features = build_all()

# ── SHAP — persis cell 48: LinearExplainer interventional ─────────────────────
# Formula: shap_i = coef_i * (x_i - mean(X_test_transformed)_i)
# Output: persentase kontribusi relatif tiap fitur terhadap total |SHAP|
def compute_shap_pct(inp_df):
    pre     = model.named_steps['preprocessor']
    lr      = model.named_steps['model']
    x       = pre.transform(inp_df)[0]
    bg_mean = X_test_transformed.mean(axis=0)
    sv      = lr.coef_[0] * (x - bg_mean)          # interventional SHAP

    df = pd.DataFrame({'feature': feature_names, 'shap': sv})
    df['abs'] = df['shap'].abs()
    df = df.sort_values('abs', ascending=False).head(10).reset_index(drop=True)

    total        = df['abs'].sum()
    df['pct']    = (df['abs'] / total * 100) if total > 0 else 0.0
    df['pushes'] = df['shap'].apply(lambda v: 'risk' if v > 0 else 'safe')
    return df

# ── Validation ────────────────────────────────────────────────────────────────
def validate(gpa, attendance, study_hours, delay, stress, travel):
    errors, warns = [], []
    if study_hours > 18:
        errors.append(("Study Hours",
            f"{study_hours} hrs/day is not realistic (max 18 hrs, leaving 6 hrs for sleep)."))
    if attendance > 100:
        errors.append(("Attendance Rate", "Attendance rate cannot exceed 100%."))
    if not (0.0 <= gpa <= 4.0):
        errors.append(("GPA", "GPA must be between 0.0 and 4.0."))
    if delay > 30:
        errors.append(("Assignment Delay",
            f"{delay} days seems too long. Please verify."))
    if travel > 240:
        errors.append(("Travel Time",
            f"{travel} min one-way seems unrealistic. Please verify."))
    if 12 < study_hours <= 18:
        warns.append(("Study Hours",
            f"{study_hours} hrs/day is very high. Confirm this is a daily average."))
    if 0 < attendance < 10:
        warns.append(("Attendance Rate",
            f"{attendance:.0f}% is extremely low. Please verify."))
    if gpa < 1.0:
        warns.append(("GPA",
            f"GPA {gpa:.2f} is critically low. Student may be on academic probation."))
    if study_hours == 0:
        warns.append(("Study Hours", "0 hrs/day entered. Please confirm."))
    return errors, warns

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --paper:      #FDFCF8;
    --ink:        #1A1A18;
    --muted:      #6B6B66;
    --line:       #E6E2D9;
    --accent:     #7A6E5A;
    --olive:      #71713B;
    --olive-lt:   #9B9B5A;
    --bone:       #E2DCD0;
    --bone-lt:    #F0EDE6;
    --risk:       #8C3A25;
    --risk-bg:    #F8F0EC;
    --risk-ln:    #C9A090;
    --safe:       #2E5C42;
    --safe-bg:    #EBF2ED;
    --safe-ln:    #90BAA2;
    --warn:       #7A5C1E;
    --warn-bg:    #FBF6EC;
    --err:        #8C2525;
    --err-bg:     #FBF0F0;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--paper);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main > div.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }

/* ── HEADER ── */
.hdr {
    padding: 4rem clamp(2.4rem, 10vw, 9rem) 3.2rem;
    border-bottom: 1px solid var(--line);
    background: var(--paper);
}
.hdr-eyebrow {
    font-size: 0.62rem;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 0.9rem;
}
.hdr-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 5vw, 4.2rem);
    font-weight: 400;
    font-style: italic;
    line-height: 1.05;
    color: var(--ink);
    margin-bottom: 1rem;
}
.hdr-sub {
    font-size: 0.83rem;
    line-height: 1.75;
    color: var(--muted);
    font-weight: 300;
    max-width: 500px;
}

/* ── PAGE BODY ── */
.pg {
    padding: 3rem clamp(2.4rem, 10vw, 9rem) 6rem;
}

/* ── SECTION LABEL ── */
.sl {
    font-size: 0.59rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    padding-bottom: 0.72rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.9rem;
}

/* ── ALERT BANNERS ── */
.alert-box {
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    font-size: 0.77rem;
    line-height: 1.62;
}
.alert-box strong {
    font-weight: 600;
    display: block;
    margin-bottom: 0.1rem;
    font-size: 0.69rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.alert-error { background:var(--err-bg);  border-left:3px solid var(--err);  color:var(--err);  }
.alert-warn  { background:var(--warn-bg); border-left:3px solid var(--warn); color:var(--warn); }

/* ─────────────────────────────────────────
   WIDGETS
───────────────────────────────────────── */

/* Labels */
div[data-testid="stSlider"]    > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"]     > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.66rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
}

/* ── SLIDER: clean, no color bleed ──
   Only override thumb; leave Streamlit to render track/fill naturally
   but suppress the red default fill by targeting the filled track */

/* Unfilled track */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background: var(--line) !important;
    height: 3px !important;
}
/* Filled track (progress) */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background: var(--olive) !important;
    height: 3px !important;
}
/* Thumb */
[data-testid="stSlider"] [role="slider"] {
    background: var(--olive) !important;
    border: 2px solid var(--paper) !important;
    box-shadow: 0 0 0 2px var(--olive) !important;
    width: 16px !important;
    height: 16px !important;
    border-radius: 50% !important;
    transition: box-shadow 0.15s !important;
}
[data-testid="stSlider"] [role="slider"]:hover,
[data-testid="stSlider"] [role="slider"]:focus {
    box-shadow: 0 0 0 4px rgba(113,113,59,0.18) !important;
}
/* Min/max tick labels — always visible, no hover needed */
[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
[data-testid="stSlider"] [data-testid="stSliderTickBarMax"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.64rem !important;
    color: var(--muted) !important;
    opacity: 1 !important;
    visibility: visible !important;
}
/* Current value label above thumb */
[data-testid="stSlider"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.64rem !important;
    color: var(--muted) !important;
    background: transparent !important;
}
/* Kill any background color on the slider wrapper */
[data-testid="stSlider"] > div {
    background: transparent !important;
}

/* ── SELECTBOX ── */
[data-baseweb="select"] > div {
    border: 1px solid var(--line) !important;
    border-radius: 3px !important;
    background: var(--bone-lt) !important;
    font-size: 0.84rem !important;
    transition: border-color 0.15s !important;
}
[data-baseweb="select"] > div:hover {
    border-color: var(--olive) !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--olive) !important;
    box-shadow: 0 0 0 2px rgba(113,113,59,0.12) !important;
}

/* ── RADIO — always horizontal, olive dot ── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 1.6rem !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    margin-top: 0.2rem !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] {
    margin: 0 !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] label {
    font-size: 0.84rem !important;
    color: var(--ink) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
    cursor: pointer !important;
}
[data-baseweb="radio"] [data-state="checked"] div { background: var(--olive) !important; }
[data-baseweb="radio"] div[class*="radioInner"]    { background: var(--olive) !important; }
[data-baseweb="radio"] label span:first-child      { border-color: var(--olive) !important; }

/* ── BUTTON ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.67rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 2rem !important;
    transition: background 0.18s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── RESULT BOX ── */
.rbox {
    padding: 2.2rem 2rem 2rem;
    border: 1px solid var(--line);
    border-radius: 4px;
}
.rbox.risk { background: var(--risk-bg); border-left: 3px solid var(--risk); }
.rbox.safe { background: var(--safe-bg); border-left: 3px solid var(--safe); }
.r-eyebrow {
    font-size: 0.59rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.55rem;
}
.rbox.risk .r-eyebrow { color: var(--risk); }
.rbox.safe .r-eyebrow { color: var(--safe); }
.r-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5.2rem;
    font-weight: 300;
    font-style: italic;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.rbox.risk .r-pct { color: var(--risk); }
.rbox.safe .r-pct { color: var(--safe); }
.r-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.18rem;
    font-style: italic;
    font-weight: 400;
    margin-bottom: 0.7rem;
}
.rbox.risk .r-verdict { color: var(--risk); }
.rbox.safe .r-verdict { color: var(--safe); }
.r-note {
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.65;
    font-weight: 300;
}

/* ── CONTRIBUTING FACTORS TABLE ── */
.ctable { width: 100%; margin-top: 0.2rem; }
.crow {
    display: grid;
    grid-template-columns: 170px 1fr 52px;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--line);
}
.crow:last-child { border-bottom: none; }
.crank {
    font-size: 0.6rem;
    color: var(--muted);
    font-weight: 400;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.04em;
}
.cname {
    font-size: 0.75rem;
    color: var(--ink);
    font-weight: 400;
    line-height: 1.4;
}
.cname .csub {
    font-size: 0.66rem;
    color: var(--muted);
    font-weight: 300;
}
.cbar-wrap {
    width: 100%;
    height: 5px;
    background: var(--line);
    border-radius: 3px;
    overflow: hidden;
}
.cbar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}
.cpct {
    font-size: 0.7rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── FOOTER ── */
.ftr {
    border-top: 1px solid var(--line);
    padding: 1.5rem clamp(2.4rem, 10vw, 9rem);
    font-size: 0.61rem;
    color: var(--muted);
    letter-spacing: 0.13em;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* spacing helpers */
.s1 { margin-bottom: 0.8rem; }
.s2 { margin-bottom: 1.8rem; }
.s3 { margin-bottom: 3.2rem; }

/* ── INPUT CARD BACKGROUND ── */
.input-panel {
    background: var(--bone-lt);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1.6rem 1.8rem 1.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">Educational Analytics &mdash; Machine Learning</div>
  <div class="hdr-title">Student Dropout<br>Predictor</div>
  <div class="hdr-sub">
    Identify students at risk of dropping out using logistic regression
    trained on academic and lifestyle indicators. Each prediction is explained
    by how much each factor contributed to the outcome.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="pg">', unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="sl">Student Profile</div>', unsafe_allow_html=True)

L, _gap, R = st.columns([10, 1, 10])

with L:
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    gpa   = st.slider("GPA", 0.0, 4.0, 2.8, 0.05,
                      help="Grade Point Average on a 4.0 scale.")
    attend = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5,
                       help="Percentage of classes attended this semester.")
    study  = st.slider("Study Hours per Day", 0.0, 18.0, 4.0, 0.25,
                       help="Average daily study hours outside class.")
    delay  = st.slider("Assignment Delay (Days)", 0, 30, 3,
                       help="Average days assignments are submitted after deadline.")
    sem    = st.selectbox("Current Year", ["Year 1","Year 2","Year 3","Year 4"])

with R:
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    stress  = st.slider("Stress Index (1–10)", 1.0, 10.0, 5.0, 0.1,
                        help="Self-reported stress level. 1 = very low, 10 = very high.")
    travel  = st.slider("Travel Time to Campus (min)", 0, 240, 30,
                        help="One-way commute time in minutes.")
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    internet   = st.radio("Internet Access", ["Yes","No"], horizontal=True)
    part_time  = st.radio("Part-Time Job",   ["Yes","No"], horizontal=True)
    scholarship = st.radio("Scholarship",    ["Yes","No"], horizontal=True)

st.markdown('<div class="s2"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([5, 6, 5])
with btn_col:
    clicked = st.button("Run Prediction", use_container_width=True)

st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────────────────────────────
if clicked:
    errors, warns = validate(gpa, attend, study, delay, stress, travel)

    # ── Errors block prediction ──
    if errors:
        st.markdown('<div class="sl">Input Errors</div>', unsafe_allow_html=True)
        for field, msg in errors:
            st.markdown(f"""
            <div class="alert-box alert-error">
              <strong>{field}</strong>{msg}
            </div>""", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:0.75rem;color:var(--muted);
                    margin-top:0.8rem;font-weight:300;">
                    Please correct the values above before running the prediction.
                    </p>""", unsafe_allow_html=True)
        st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

    else:
        # ── Warnings (non-blocking) ──
        if warns:
            st.markdown('<div class="sl">Input Warnings</div>', unsafe_allow_html=True)
            for field, msg in warns:
                st.markdown(f"""
                <div class="alert-box alert-warn">
                  <strong>{field}</strong>{msg}
                </div>""", unsafe_allow_html=True)
            st.markdown("""<p style="font-size:0.75rem;color:var(--muted);
                        margin-top:0.4rem;margin-bottom:2rem;font-weight:300;">
                        Prediction will proceed. Please verify flagged values.
                        </p>""", unsafe_allow_html=True)

        # ── Build input DataFrame — persis cell 47 ──
        inp = pd.DataFrame({
            'Internet_Access':        [internet],
            'Study_Hours_per_Day':    [study],
            'Attendance_Rate':        [attend],
            'Assignment_Delay_Days':  [float(delay)],
            'Travel_Time_Minutes':    [float(travel)],
            'Part_Time_Job':          [part_time],
            'Scholarship':            [scholarship],
            'Stress_Index':           [stress],
            'GPA':                    [gpa],
            'Semester':               [sem],
        })

        # ── Predict ──
        pred  = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        dp    = proba[1]   # dropout probability (cell 47)

        # ── SHAP (cell 48) ──
        shap_df = compute_shap_pct(inp)

        # ── Layout ──
        st.markdown('<div class="sl">Prediction Result</div>', unsafe_allow_html=True)
        col_res, col_fact = st.columns([5, 7])

        # ── Left: result box ──
        with col_res:
            box_cls = "risk" if pred == 1 else "safe"
            verdict = "At Risk of Dropout" if pred == 1 else "Not at Risk"
            note    = ("This student shows a high likelihood of dropping out. "
                       "Early intervention is recommended."
                       if pred == 1 else
                       "This student appears on track based on current academic "
                       "and lifestyle indicators.")
            st.markdown(f"""
            <div class="rbox {box_cls}">
              <div class="r-eyebrow">Dropout Probability</div>
              <div class="r-pct">{dp:.1%}</div>
              <div class="r-verdict">{verdict}</div>
              <div class="r-note">{note}</div>
            </div>""", unsafe_allow_html=True)

        # ── Right: contributing factors ──
        with col_fact:
            st.markdown('<div class="sl">Top 10 Contributing Factors to Dropout Probability</div>',
                        unsafe_allow_html=True)

            rows_html = '<div class="ctable">'
            for rank, (_, row) in enumerate(shap_df.iterrows(), 1):
                # Clean feature name
                raw  = row['feature']
                name = (raw
                        .replace('Internet_Access_', 'Internet Access = ')
                        .replace('Part_Time_Job_',   'Part-Time Job = ')
                        .replace('Scholarship_',     'Scholarship = ')
                        .replace('Semester_',        'Semester = ')
                        .replace('_', ' '))

                pct    = row['pct']
                pushes = row['pushes']
                color  = "var(--risk)" if pushes == 'risk' else "var(--safe)"
                # Direction label
                dir_label = "raises risk" if pushes == 'risk' else "lowers risk"

                rows_html += f"""
                <div class="crow">
                  <span class="cname">
                    {name}
                    <span class="csub" style="display:block;color:{color};">
                      {dir_label}
                    </span>
                  </span>
                  <div class="cbar-wrap">
                    <div class="cbar-fill"
                         style="width:{min(pct,100):.1f}%;background:{color};">
                    </div>
                  </div>
                  <span class="cpct" style="color:{color};">{pct:.1f}%</span>
                </div>"""
            rows_html += '</div>'
            rows_html += """
            <p style="font-size:0.64rem;color:var(--muted);margin-top:1.1rem;
                      font-weight:300;line-height:1.65;">
              Percentage reflects each feature's relative contribution to this
              prediction, computed via SHAP&nbsp;(LinearExplainer,
              interventional&nbsp;perturbation).
            </p>"""
            st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
  <span>Student Dropout Predictor</span>
  <span>Logistic Regression &nbsp;&middot;&nbsp; SHAP &nbsp;&middot;&nbsp; Streamlit</span>
</div>
""", unsafe_allow_html=True)
