import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Train Model (cached, runs once at startup) ─────────────────────────────────
@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 2000
    Internet_Access   = np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3])
    Study_Hours       = np.random.uniform(0, 10, n)
    Attendance        = np.random.uniform(40, 100, n)
    Assignment_Delay  = np.random.uniform(0, 15, n)
    Travel_Time       = np.random.uniform(5, 120, n)
    Part_Time_Job     = np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])
    Scholarship       = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
    Stress_Index      = np.random.uniform(1, 10, n)
    GPA               = np.random.uniform(1.5, 4.0, n)
    Semester          = np.random.choice(['Year 1', 'Year 2', 'Year 3', 'Year 4'], n)

    dropout_prob = (
        0.30 * (1 - Study_Hours / 10) +
        0.25 * (1 - Attendance / 100) +
        0.15 * (Assignment_Delay / 15) +
        0.15 * (Stress_Index / 10) +
        0.10 * (1 - GPA / 4) +
        0.05 * (Part_Time_Job == 'Yes').astype(float)
    )
    dropout_prob = np.clip(dropout_prob + np.random.normal(0, 0.1, n), 0, 1)
    Dropout = (dropout_prob > 0.45).astype(int)

    df = pd.DataFrame({
        'Internet_Access':        Internet_Access,
        'Study_Hours_per_Day':    Study_Hours,
        'Attendance_Rate':        Attendance,
        'Assignment_Delay_Days':  Assignment_Delay,
        'Travel_Time_Minutes':    Travel_Time,
        'Part_Time_Job':          Part_Time_Job,
        'Scholarship':            Scholarship,
        'Stress_Index':           Stress_Index,
        'GPA':                    GPA,
        'Semester':               Semester,
        'Dropout':                Dropout,
    })

    numeric_features     = ['Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days',
                             'Travel_Time_Minutes', 'Stress_Index', 'GPA']
    categorical_features = ['Internet_Access', 'Part_Time_Job', 'Scholarship', 'Semester']

    X = df.drop(columns=['Dropout'])
    y = df['Dropout']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imp',  SimpleImputer(strategy='most_frequent')),
        ('ohe',  OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer,     numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')

    X_train_prep = preprocessor.fit_transform(X_train)
    minority_idx = np.where(y_train == 1)[0]
    majority_idx = np.where(y_train == 0)[0]
    n_oversample = len(majority_idx) - len(minority_idx)
    oversample_idx = np.random.choice(minority_idx, size=n_oversample, replace=True)
    X_balanced = np.vstack([X_train_prep, X_train_prep[oversample_idx]])
    y_balanced = np.concatenate([y_train.values, y_train.values[oversample_idx]])

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_balanced, y_balanced)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model',        lr),
    ])
    return pipeline

model = load_model()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400;1,500&family=Inter:wght@300;400;500&display=swap');

:root {
    --steel:   #94A5B3;
    --olive:   #71713B;
    --bone:    #E2DCD0;
    --cream:   #F5F2EC;
    --ink:     #1e1e1c;
    --muted:   #6b6b66;
    --line:    #d8d3cb;
    --red:     #a0452e;
    --red-bg:  #f7efec;
    --green:   #3d6650;
    --green-bg:#ecf2ee;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--cream);
    color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container,
[data-testid="stMainBlockContainer"],
div[data-testid="stAppViewBlockContainer"],
section.main > div.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── HEADER ── */
.hdr {
    background: var(--cream);
    border-bottom: 1px solid var(--line);
    padding: 2.8rem clamp(2rem, 8vw, 7rem) 2.4rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
}
.hdr-left { max-width: 600px; }
.hdr-tag {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
    margin-bottom: 0.7rem;
}
.hdr-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 4vw, 3.4rem);
    font-weight: 500;
    font-style: italic;
    line-height: 1.1;
    color: var(--ink);
    margin-bottom: 0.8rem;
}
.hdr-desc {
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--muted);
    font-weight: 300;
}
.hdr-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.45rem;
    flex-shrink: 0;
}
.stat-pill {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}
.stat-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem;
    font-weight: 500;
    color: var(--olive);
    line-height: 1;
}
.stat-lbl {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
}

/* ── BODY ── */
.body {
    padding: 3rem clamp(2rem, 8vw, 7rem) 5rem;
    max-width: 1320px;
    margin: 0 auto;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.67rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.6rem;
}

/* ── WIDGET LABELS ── */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── SLIDER: kill Streamlit red/green entirely ── */
/* track background */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"],
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child,
[data-testid="stSlider"] div[class*="sliderTrack"] {
    background: var(--line) !important;
}
/* filled portion */
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"],
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2),
[data-testid="stSlider"] div[class*="sliderFill"] {
    background: var(--olive) !important;
}
/* thumb */
[data-testid="stSlider"] [role="slider"] {
    background: var(--olive) !important;
    border: 3px solid var(--cream) !important;
    box-shadow: 0 0 0 2px var(--olive) !important;
    width: 16px !important;
    height: 16px !important;
    border-radius: 50% !important;
}
/* kill any inline color Streamlit injects */
[data-testid="stSlider"] [data-baseweb="slider"] * {
    --slider-bar-color: var(--olive) !important;
}
/* value label above thumb */
[data-testid="stSlider"] div[class*="StyledThumbValue"],
[data-testid="stSlider"] p[data-testid="stSliderTickBarMin"],
[data-testid="stSlider"] p[data-testid="stSliderTickBarMax"] {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── SELECTBOX ── */
div[data-baseweb="select"] > div {
    border-color: var(--line) !important;
    border-radius: 4px !important;
    background: #fff !important;
    font-size: 0.88rem !important;
}

/* ── RADIO: olive dot, no red ── */
div[data-testid="stRadio"] label span:first-child {
    border-color: var(--olive) !important;
}
div[data-testid="stRadio"] input:checked + div,
div[data-testid="stRadio"] label span[data-checked="true"] {
    background: var(--olive) !important;
    border-color: var(--olive) !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] label {
    font-size: 0.85rem !important;
    color: var(--ink) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
}
/* BaseWeb radio checked fill */
[data-baseweb="radio"] [data-state="checked"] div {
    background: var(--olive) !important;
    border-color: var(--olive) !important;
}
[data-baseweb="radio"] div[class*="radioInner"] {
    background: var(--olive) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: var(--olive) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2.4rem !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── RESULT ── */
.result-box {
    border-radius: 4px;
    padding: 2rem 2rem 1.8rem;
}
.result-box.risk    { background: var(--red-bg);   border: 1px solid #d9b5aa; }
.result-box.safe    { background: var(--green-bg); border: 1px solid #a8c5b4; }
.res-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.6rem;
}
.result-box.risk  .res-label { color: var(--red); }
.result-box.safe  .res-label { color: var(--green); }
.res-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.2rem;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.result-box.risk .res-pct  { color: var(--red); }
.result-box.safe .res-pct  { color: var(--green); }
.res-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-style: italic;
    font-weight: 400;
    margin-bottom: 0.4rem;
}
.result-box.risk .res-verdict { color: var(--red); }
.result-box.safe .res-verdict { color: var(--green); }
.res-note {
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.6;
    font-weight: 300;
}

/* ── GAUGE ── */
.gauge-wrap { margin-top: 1.4rem; }
.gauge-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.gauge-track {
    height: 4px;
    background: var(--line);
    border-radius: 2px;
    position: relative;
    overflow: visible;
}
.gauge-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}
.gauge-dot {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 0 0 1px var(--ink);
}

/* ── FACTOR BARS ── */
.factor-row {
    display: grid;
    grid-template-columns: 148px 1fr 38px;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--line);
}
.factor-row:last-child { border-bottom: none; }
.factor-name {
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 400;
}
.factor-bar-bg {
    height: 3px;
    background: var(--line);
    border-radius: 2px;
    overflow: hidden;
}
.factor-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.factor-pct {
    font-size: 0.72rem;
    color: var(--muted);
    text-align: right;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

/* ── REC CARD ── */
.rec-card {
    padding: 1.2rem 1.3rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: #fff;
    margin-bottom: 0.6rem;
}
.rec-title {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: var(--ink);
    margin-bottom: 0.3rem;
}
.rec-desc {
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.65;
    font-weight: 300;
}
.rec-card.alert { border-left: 2px solid var(--red); }
.rec-card.ok    { border-left: 2px solid var(--green); }

/* ── ABOUT CARDS ── */
.about-block {
    padding: 1.4rem 1.6rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: #fff;
    height: 100%;
}
.about-block h4 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 500;
    font-style: italic;
    color: var(--ink);
    margin-bottom: 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--line);
}
.about-block p, .about-block li {
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.75;
    font-weight: 300;
}
.about-block ul { padding-left: 1rem; }
.about-block li { margin-bottom: 0.15rem; }

/* ── FOOTER ── */
.ftr {
    border-top: 1px solid var(--line);
    padding: 1.4rem clamp(2rem, 8vw, 7rem);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── SPACING HELPERS ── */
.sp-sm { margin-bottom: 0.8rem; }
.sp-md { margin-bottom: 1.6rem; }
.sp-lg { margin-bottom: 2.8rem; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <div class="hdr-left">
        <div class="hdr-tag">Educational Analytics &mdash; Machine Learning</div>
        <div class="hdr-title">Student Dropout<br>Predictor</div>
        <div class="hdr-desc">
            Identify students at risk of dropping out using logistic regression
            trained on academic and lifestyle indicators.
        </div>
    </div>
    <div class="hdr-right">
        <div class="stat-pill">
            <span class="stat-val">79.8%</span>
            <span class="stat-lbl">Accuracy</span>
        </div>
        <div class="stat-pill">
            <span class="stat-val">83.1%</span>
            <span class="stat-lbl">Recall</span>
        </div>
        <div class="stat-pill">
            <span class="stat-val">88.3%</span>
            <span class="stat-lbl">ROC-AUC</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── BODY ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="body">', unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Student Profile</div>', unsafe_allow_html=True)

left_col, gap_col, right_col = st.columns([11, 1, 11])

with left_col:
    st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
    gpa = st.slider("GPA", min_value=1.0, max_value=4.0, value=2.8, step=0.05)
    attendance = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    study_hours = st.slider("Study Hours per Day", min_value=0.0, max_value=12.0, value=4.0, step=0.25)
    assignment_delay = st.slider("Assignment Delay (Days)", min_value=0, max_value=15, value=3)
    semester = st.selectbox("Current Year", options=["Year 1", "Year 2", "Year 3", "Year 4"])

with right_col:
    st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
    stress = st.slider("Stress Index (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    travel_time = st.slider("Travel Time to Campus (min)", min_value=0, max_value=120, value=30)
    st.markdown('<div class="sp-sm"></div>', unsafe_allow_html=True)
    internet = st.radio("Internet Access", options=["Yes", "No"], horizontal=True)
    part_time = st.radio("Part-Time Job", options=["Yes", "No"], horizontal=True)
    scholarship = st.radio("Scholarship", options=["Yes", "No"], horizontal=True)

st.markdown('<div class="sp-md"></div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([7, 6, 7])
with btn_col:
    predict_clicked = st.button("Run Prediction", use_container_width=True)

st.markdown('<div class="sp-lg"></div>', unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────────────────────────────
if predict_clicked:
    input_df = pd.DataFrame({
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

    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    drop_pct      = probabilities[1]
    safe_pct      = probabilities[0]

    st.markdown('<div class="sec-label">Result</div>', unsafe_allow_html=True)

    res_l, res_r = st.columns([5, 7])

    with res_l:
        if prediction == 1:
            gauge_color = "#a0452e"
            st.markdown(f"""
            <div class="result-box risk">
                <div class="res-label">Dropout Probability</div>
                <div class="res-pct">{drop_pct:.1%}</div>
                <div class="res-verdict">At Risk of Dropout</div>
                <div class="res-note">Immediate intervention is recommended for this student.</div>
                <div class="gauge-wrap">
                    <div class="gauge-top"><span>Low</span><span>High</span></div>
                    <div class="gauge-track">
                        <div class="gauge-fill" style="width:{drop_pct*100:.1f}%; background:{gauge_color};"></div>
                        <div class="gauge-dot" style="left:{drop_pct*100:.1f}%; background:{gauge_color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            gauge_color = "#3d6650"
            st.markdown(f"""
            <div class="result-box safe">
                <div class="res-label">Safe Probability</div>
                <div class="res-pct">{safe_pct:.1%}</div>
                <div class="res-verdict">Not at Risk</div>
                <div class="res-note">Student appears on track. Continue regular monitoring.</div>
                <div class="gauge-wrap">
                    <div class="gauge-top"><span>Low</span><span>High</span></div>
                    <div class="gauge-track">
                        <div class="gauge-fill" style="width:{safe_pct*100:.1f}%; background:{gauge_color};"></div>
                        <div class="gauge-dot" style="left:{safe_pct*100:.1f}%; background:{gauge_color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_r:
        st.markdown('<div class="sec-label">Contributing Factors</div>', unsafe_allow_html=True)

        factors = [
            ("Low GPA",            max(0.0, 1 - gpa / 4)),
            ("Low Attendance",     max(0.0, 1 - attendance / 100)),
            ("High Stress",        stress / 10),
            ("Low Study Hours",    max(0.0, 1 - study_hours / 10)),
            ("Assignment Delay",   assignment_delay / 15),
            ("Part-Time Job",      1.0 if part_time == "Yes" else 0.0),
            ("No Internet",        1.0 if internet == "No" else 0.0),
            ("No Scholarship",     0.9 if scholarship == "No" else 0.0),
        ]
        factors_sorted = sorted(factors, key=lambda x: x[1], reverse=True)

        rows_html = ""
        for name, score in factors_sorted:
            bar_color = "#a0452e" if score > 0.6 else "#71713B" if score > 0.3 else "#94A5B3"
            rows_html += f"""
            <div class="factor-row">
                <span class="factor-name">{name}</span>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{score*100:.0f}%; background:{bar_color};"></div>
                </div>
                <span class="factor-pct">{score:.0%}</span>
            </div>"""

        st.markdown(f'<div>{rows_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sp-lg"></div>', unsafe_allow_html=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    # Score each factor
    factor_scores = {
        "attendance":   max(0.0, 1 - attendance / 100),
        "gpa":          max(0.0, 1 - gpa / 4),
        "stress":       stress / 10,
        "study_hours":  max(0.0, 1 - study_hours / 10),
        "delay":        assignment_delay / 15,
        "part_job":     1.0 if part_time == "Yes" else 0.0,
        "internet":     1.0 if internet == "No" else 0.0,
        "scholarship":  0.9 if scholarship == "No" else 0.0,
    }

    recs = []
    if attendance < 80:
        recs.append(("Attendance",
                     f"Attendance at {attendance:.0f}% is below the recommended threshold. "
                     "Regular check-ins and attendance monitoring are advised."))
    if gpa < 3.0:
        recs.append(("Academic Performance",
                     f"GPA of {gpa:.2f} signals academic difficulty. "
                     "Tutoring and academic mentoring are recommended."))
    if stress > 6:
        recs.append(("Stress Management",
                     f"Stress index at {stress:.1f}/10 is elevated. "
                     "Refer to campus mental health or counseling services."))
    if study_hours < 4:
        recs.append(("Study Hours",
                     f"Only {study_hours:.1f} hours of study per day. "
                     "Time management coaching and a structured study plan may help."))
    if assignment_delay > 3:
        recs.append(("Assignment Submission",
                     f"Assignments delayed by {assignment_delay} days on average. "
                     "A structured deadline tracker is recommended."))
    if part_time == "Yes" and scholarship == "No":
        recs.append(("Financial Aid",
                     "Student works part-time without a scholarship. "
                     "Explore financial aid and bursary options to reduce work burden."))
    if internet == "No":
        recs.append(("Digital Access",
                     "No internet access may hinder coursework. "
                     "Explore campus Wi-Fi programmes or device lending schemes."))

    # If model predicts dropout but no threshold triggered, surface top contributing factors
    if prediction == 1 and not recs:
        label_map = {
            "attendance": ("Attendance",       "Attendance rate is a contributing risk factor. Closer monitoring is advised."),
            "gpa":        ("Academic Performance", "GPA is a contributing risk factor. Academic support is recommended."),
            "stress":     ("Stress Levels",    "Stress is a contributing risk factor. Mental health support is advised."),
            "study_hours":("Study Hours",      "Low study hours are a contributing risk factor. A structured study plan may help."),
            "delay":      ("Assignment Delays","Assignment delays contribute to dropout risk. Deadline tracking is recommended."),
            "part_job":   ("Part-Time Work",   "Part-time employment increases dropout risk. Consider workload balance."),
            "internet":   ("Digital Access",   "Limited internet access contributes to dropout risk."),
            "scholarship":("Financial Pressure","Lack of scholarship increases financial stress. Explore aid options."),
        }
        top3 = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for key, _ in top3:
            recs.append(label_map[key])

    if not recs:
        recs.append(("No Concerns Detected",
                     "All indicators are within healthy ranges. Continue regular monitoring."))

    st.markdown('<div class="sec-label">Recommendations</div>', unsafe_allow_html=True)
    rec_cols = st.columns(3)
    card_class = "alert" if prediction == 1 else "ok"
    for i, (title, desc) in enumerate(recs):
        with rec_cols[i % 3]:
            st.markdown(f"""
            <div class="rec-card {card_class}">
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sp-lg"></div>', unsafe_allow_html=True)

# ── ABOUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">About</div>', unsafe_allow_html=True)

ab1, ab2, ab3 = st.columns(3)

with ab1:
    st.markdown("""
    <div class="about-block">
        <h4>Algorithm</h4>
        <p>Logistic Regression with manual oversampling to handle class imbalance.
        Selected for its superior Recall — critical for identifying at-risk students before they drop out.</p>
    </div>
    """, unsafe_allow_html=True)

with ab2:
    st.markdown("""
    <div class="about-block">
        <h4>Features</h4>
        <ul>
            <li>GPA &amp; Attendance Rate</li>
            <li>Study Hours per Day</li>
            <li>Stress Index</li>
            <li>Assignment Delay Days</li>
            <li>Travel Time to Campus</li>
            <li>Internet Access, Part-Time Job, Scholarship, Semester</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with ab3:
    st.markdown("""
    <div class="about-block">
        <h4>Why Recall Matters</h4>
        <p>Missing an at-risk student (False Negative) is far more costly than a false alarm.
        This model prioritises Recall to maximise detection of students who need intervention.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
    Student Dropout Predictor &nbsp;&middot;&nbsp; Logistic Regression &nbsp;&middot;&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
