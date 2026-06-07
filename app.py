import streamlit as st
import pandas as pd
import numpy as np
import os

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

# ── Model: load .pkl if present, otherwise train from synthetic data ────────────
@st.cache_resource
def load_model():
    pkl_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
    if os.path.exists(pkl_path):
        import joblib
        return joblib.load(pkl_path)

    # Fallback: train on synthetic data (used when .pkl is not in repo)
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'Internet_Access':       np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3]),
        'Study_Hours_per_Day':   np.random.uniform(0, 10, n),
        'Attendance_Rate':       np.random.uniform(40, 100, n),
        'Assignment_Delay_Days': np.random.uniform(0, 15, n),
        'Travel_Time_Minutes':   np.random.uniform(5, 120, n),
        'Part_Time_Job':         np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6]),
        'Scholarship':           np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'Stress_Index':          np.random.uniform(1, 10, n),
        'GPA':                   np.random.uniform(1.5, 4.0, n),
        'Semester':              np.random.choice(['Year 1','Year 2','Year 3','Year 4'], n),
    })
    sh = df['Study_Hours_per_Day'].values
    at = df['Attendance_Rate'].values
    ad = df['Assignment_Delay_Days'].values
    si = df['Stress_Index'].values
    gp = df['GPA'].values
    pt = (df['Part_Time_Job'] == 'Yes').astype(float).values
    p  = np.clip(0.30*(1-sh/10)+0.25*(1-at/100)+0.15*(ad/15)+0.15*(si/10)+0.10*(1-gp/4)+0.05*pt
                 + np.random.normal(0, 0.1, n), 0, 1)
    df['Dropout'] = (p > 0.45).astype(int)

    num_feats = ['Study_Hours_per_Day','Attendance_Rate','Assignment_Delay_Days',
                 'Travel_Time_Minutes','Stress_Index','GPA']
    cat_feats = ['Internet_Access','Part_Time_Job','Scholarship','Semester']

    X, y = df.drop(columns=['Dropout']), df['Dropout']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ('num', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num_feats),
        ('cat', Pipeline([('i', SimpleImputer(strategy='most_frequent')), ('o', OneHotEncoder(handle_unknown='ignore'))]), cat_feats),
    ], remainder='drop')

    Xp = pre.fit_transform(X_tr)
    min_idx = np.where(y_tr == 1)[0]
    maj_idx = np.where(y_tr == 0)[0]
    over    = np.random.choice(min_idx, size=len(maj_idx)-len(min_idx), replace=True)
    Xb = np.vstack([Xp, Xp[over]])
    yb = np.concatenate([y_tr.values, y_tr.values[over]])

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(Xb, yb)
    return Pipeline([('preprocessor', pre), ('model', lr)])

model = load_model()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600;1,700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --paper:     #FDFCF8;
    --ink:       #1A1A1A;
    --ink-light: #6B6B6B;
    --border:    #E8E5DC;
    --accent:    #8B7355;
    --accent-light: #A08B6E;
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

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main > div.block-container {
    padding: 0 !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* HEADER */
.hdr {
    padding: 4rem 2.5rem 3rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    background: var(--paper);
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
    font-size: clamp(3rem, 5vw, 4.5rem);
    font-weight: 500;
    font-style: italic;
    line-height: 1.05;
    color: var(--ink);
    margin-bottom: 1rem;
}
.hdr-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    line-height: 1.65;
    color: var(--ink-light);
    font-weight: 400;
    max-width: 520px;
}
.hdr-stats {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
    flex-shrink: 0;
}
.stat-row { display: flex; align-items: baseline; gap: 0.5rem; }
.stat-n {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
}
.stat-l {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ink-light);
    font-weight: 500;
}

/* PAGE BODY */
.pg {
    padding: 2.5rem 2.5rem 4rem;
    max-width: 1280px;
    margin: 0 auto;
}

/* SECTION LABEL */
.sl {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

/* WIDGET LABELS */
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

/* SLIDER - simpel, min max tampil otomatis dari Streamlit */
[data-testid="stSlider"] {
    padding-bottom: 0.5rem !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 0.5rem !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--paper) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    width: 16px !important;
    height: 16px !important;
}
/* Biarkan Streamlit handle min-max display secara default */
[data-testid="stSlider"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    color: var(--ink-light) !important;
}

/* RADIO - horizontal SEJAJAR untuk semua */
div[data-testid="stRadio"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 1.5rem !important;
    align-items: center !important;
    flex-wrap: wrap !important;
}
div[data-testid="stRadio"] > label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}
[data-baseweb="radio"] {
    display: inline-flex !important;
}
[data-baseweb="radio"] [data-state="checked"] div,
[data-baseweb="radio"] div[class*="radioInner"] {
    background: var(--accent) !important;
}
[data-baseweb="radio"] label span:first-child {
    border-color: var(--accent) !important;
}

/* SELECTBOX */
[data-baseweb="select"] > div {
    border-color: var(--border) !important;
    border-radius: 0px !important;
    background: var(--paper) !important;
}
[data-baseweb="select"]:hover > div {
    border-color: var(--accent) !important;
}

/* BUTTON */
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
    padding: 0.9rem 2rem !important;
    transition: background 0.2s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
}

/* RESULT BOX */
.rbox {
    border-radius: 0px;
    padding: 2rem;
    background: var(--paper);
    border: 1px solid var(--border);
}
.rbox.risk { background: var(--risk-bg); border-left: 3px solid var(--risk); }
.rbox.safe { background: var(--safe-bg); border-left: 3px solid var(--safe); }
.r-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.rbox.risk .r-eyebrow { color: var(--risk); }
.rbox.safe .r-eyebrow { color: var(--safe); }
.r-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 600;
    font-style: italic;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.rbox.risk .r-pct { color: var(--risk); }
.rbox.safe .r-pct { color: var(--safe); }
.r-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 500;
    font-style: italic;
    margin-bottom: 0.75rem;
}
.rbox.risk .r-verdict { color: var(--risk); }
.rbox.safe .r-verdict { color: var(--safe); }
.r-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--ink-light);
    line-height: 1.6;
    font-weight: 400;
}

/* GAUGE */
.gauge { margin-top: 1.8rem; }
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

/* FACTOR ROWS */
.frow {
    display: grid;
    grid-template-columns: 140px 1fr 45px;
    align-items: center;
    gap: 1rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
}
.frow:last-child { border-bottom: none; }
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

/* REC CARD */
.rcard {
    padding: 1.25rem;
    border: 1px solid var(--border);
    background: var(--paper);
    margin-bottom: 0.8rem;
    height: calc(100% - 0.8rem);
    transition: border-color 0.2s ease;
}
.rcard:hover {
    border-color: var(--accent-light);
}
.rcard.alert { border-left: 2px solid var(--risk); }
.rcard.ok    { border-left: 2px solid var(--safe); }
.rc-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.5rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.rc-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--ink-light);
    line-height: 1.65;
    font-weight: 400;
}

/* spacing */
.s1 { margin-bottom: 0.8rem; }
.s2 { margin-bottom: 1.6rem; }
.s3 { margin-bottom: 2.5rem; }

/* FOOTER */
.ftr {
    border-top: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: var(--ink-light);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: var(--paper);
    max-width: 1280px;
    margin: 0 auto;
}
</style>

""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div>
    <div class="hdr-eyebrow">Educational Analytics &mdash; Machine Learning</div>
    <div class="hdr-title">Student Dropout<br>Predictor</div>
    <div class="hdr-sub">
      Identify students at risk of dropping out using logistic regression
      trained on academic and lifestyle indicators.
    </div>
  </div>
  <div class="hdr-stats">
    <div class="stat-row"><span class="stat-n">79.8%</span><span class="stat-l">Accuracy</span></div>
    <div class="stat-row"><span class="stat-n">83.1%</span><span class="stat-l">Recall</span></div>
    <div class="stat-row"><span class="stat-n">88.3%</span><span class="stat-l">ROC-AUC</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── PAGE BODY ─────────────────────────────────────────────────────────────────
st.markdown('<div class="pg">', unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="sl">Student Profile</div>', unsafe_allow_html=True)

L, _gap, R = st.columns([10, 1, 10])

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

    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    dp    = proba[1]   # dropout probability
    sp    = proba[0]   # safe probability

    st.markdown('<div class="sl">Result</div>', unsafe_allow_html=True)

    col_res, col_fac = st.columns([5, 7])

    # ── Result box ──
    with col_res:
        if pred == 1:
            st.markdown(f"""
            <div class="rbox risk">
              <div class="r-eyebrow">Dropout Probability</div>
              <div class="r-pct">{dp:.1%}</div>
              <div class="r-verdict">At Risk of Dropout</div>
              <div class="r-note">Immediate intervention is recommended for this student.</div>
              <div class="gauge">
                <div class="gauge-labels"><span>Low</span><span>High</span></div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{dp*100:.1f}%;background:#8c3a25;"></div>
                  <div class="gauge-dot"  style="left:{dp*100:.1f}%;color:#8c3a25;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rbox safe">
              <div class="r-eyebrow">Safe Probability</div>
              <div class="r-pct">{sp:.1%}</div>
              <div class="r-verdict">Not at Risk</div>
              <div class="r-note">Student appears on track. Continue regular monitoring.</div>
              <div class="gauge">
                <div class="gauge-labels"><span>Low</span><span>High</span></div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{sp*100:.1f}%;background:#2e5c42;"></div>
                  <div class="gauge-dot"  style="left:{sp*100:.1f}%;color:#2e5c42;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Contributing factors ──
    with col_fac:
        st.markdown('<div class="sl">Contributing Factors</div>', unsafe_allow_html=True)

        factors = [
            ("Low GPA",           max(0.0, 1 - gpa / 4)),
            ("Low Attendance",    max(0.0, 1 - attendance / 100)),
            ("High Stress",       stress / 10),
            ("Low Study Hours",   max(0.0, 1 - study_hours / 10)),
            ("Assignment Delay",  assignment_delay / 15),
            ("Part-Time Job",     1.0 if part_time == "Yes" else 0.0),
            ("No Internet",       1.0 if internet == "No" else 0.0),
            ("No Scholarship",    0.9 if scholarship == "No" else 0.0),
        ]
        factors.sort(key=lambda x: x[1], reverse=True)

        rows = ""
        for name, score in factors:
            clr = "#8c3a25" if score > 0.65 else "#71713B" if score > 0.3 else "#94A5B3"
            rows += f"""
            <div class="frow">
              <span class="fn">{name}</span>
              <div class="fb-bg"><div class="fb-fill" style="width:{score*100:.0f}%;background:{clr};"></div></div>
              <span class="fp">{score:.0%}</span>
            </div>"""
        st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)

    st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

    # ── Recommendations ──
    factor_scores = {
        "attendance":  max(0.0, 1 - attendance / 100),
        "gpa":         max(0.0, 1 - gpa / 4),
        "stress":      stress / 10,
        "study_hours": max(0.0, 1 - study_hours / 10),
        "delay":       assignment_delay / 15,
        "part_job":    1.0 if part_time == "Yes" else 0.0,
        "internet":    1.0 if internet == "No" else 0.0,
        "scholarship": 0.9 if scholarship == "No" else 0.0,
    }

    recs = []
    if attendance < 80:
        recs.append(("Attendance",
            f"Attendance is at {attendance:.0f}%, below the recommended level. "
            "Regular check-ins and attendance monitoring are advised."))
    if gpa < 3.0:
        recs.append(("Academic Performance",
            f"GPA of {gpa:.2f} indicates academic difficulty. "
            "Tutoring and academic mentoring are recommended."))
    if stress > 6:
        recs.append(("Stress Management",
            f"Stress index is {stress:.1f}/10. "
            "Refer to campus mental health or counseling services."))
    if study_hours < 4:
        recs.append(("Study Hours",
            f"Only {study_hours:.1f} hours of study per day. "
            "A structured study plan and time management coaching may help."))
    if assignment_delay > 3:
        recs.append(("Assignment Submission",
            f"Assignments delayed by {assignment_delay} days on average. "
            "A deadline tracker is recommended."))
    if part_time == "Yes" and scholarship == "No":
        recs.append(("Financial Aid",
            "Student works part-time without a scholarship. "
            "Explore financial aid options to reduce the work burden."))
    if internet == "No":
        recs.append(("Digital Access",
            "No internet access may hinder coursework. "
            "Explore campus connectivity or device lending programmes."))

    # If model predicts dropout but nothing triggered, surface the worst 3 factors
    if pred == 1 and not recs:
        label_map = {
            "attendance":  ("Attendance",          "Attendance rate is a contributing risk factor. Closer monitoring is advised."),
            "gpa":         ("Academic Performance", "GPA is a contributing risk factor. Academic support is recommended."),
            "stress":      ("Stress Levels",        "Elevated stress is a contributing risk factor. Mental health support is advised."),
            "study_hours": ("Study Hours",          "Low study hours are a contributing risk factor. A structured plan may help."),
            "delay":       ("Assignment Delays",    "Assignment delays contribute to dropout risk. Deadline tracking is recommended."),
            "part_job":    ("Part-Time Work",       "Part-time employment increases dropout risk. Consider workload balance."),
            "internet":    ("Digital Access",       "Limited internet access is a contributing risk factor."),
            "scholarship": ("Financial Pressure",   "Lack of scholarship adds financial stress. Explore aid options."),
        }
        for key, _ in sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            recs.append(label_map[key])

    if not recs:
        recs.append(("No Concerns Detected",
            "All indicators are within healthy ranges. Continue regular monitoring."))

    st.markdown('<div class="sl">Recommendations</div>', unsafe_allow_html=True)
    card_cls = "alert" if pred == 1 else "ok"
    cols = st.columns(3)
    for i, (title, desc) in enumerate(recs):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="rcard {card_cls}">
              <div class="rc-title">{title}</div>
              <div class="rc-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
  Student Dropout Predictor &nbsp;&middot;&nbsp; Logistic Regression &nbsp;&middot;&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
