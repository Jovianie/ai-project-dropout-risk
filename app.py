import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --cool-steel:  #94A5B3;
    --steel-light: #b8c8d4;
    --steel-dark:  #6b8494;
    --olive:       #71713B;
    --olive-light: #8e8e50;
    --olive-dark:  #555530;
    --bone:        #E2DCD0;
    --bone-dark:   #cfc8ba;
    --cream:       #F7F4EF;
    --ink:         #2C2C2A;
    --ink-soft:    #4a4a46;
    --warn-red:    #b85c42;
    --warn-light:  #f4ede9;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
    background-color: var(--cream);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero banner ── */
.hero {
    background: var(--cool-steel);
    padding: 3.5rem 4rem 3rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: var(--steel-light);
    border-radius: 50%;
    opacity: 0.35;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 35%;
    width: 180px; height: 180px;
    background: var(--steel-dark);
    border-radius: 50%;
    opacity: 0.2;
}
.hero-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--bone);
    opacity: 0.85;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    font-style: italic;
    color: #fff;
    line-height: 1.15;
    margin: 0 0 0.75rem;
}
.hero-sub {
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    color: var(--bone);
    opacity: 0.9;
    max-width: 560px;
    line-height: 1.65;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 2rem;
    padding: 0.28rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: #fff;
    margin-top: 1.2rem;
    backdrop-filter: blur(4px);
}

/* ── Main content wrapper ── */
.main-wrap {
    padding: 2.5rem 4rem 4rem;
    max-width: 1280px;
    margin: 0 auto;
}

/* ── Section title ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 0.3rem;
}
.section-sub {
    font-family: 'Lora', serif;
    font-size: 0.9rem;
    font-style: italic;
    color: var(--ink-soft);
    margin: 0 0 1.5rem;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, var(--olive) 0%, var(--bone-dark) 60%, transparent 100%);
    border: none;
    margin: 1.2rem 0 2rem;
}

/* ── Cards ── */
.card {
    background: #fff;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    border: 1px solid var(--bone-dark);
    box-shadow: 0 2px 12px rgba(44,44,42,0.06);
    height: 100%;
}
.card-olive {
    background: var(--olive);
    border-color: var(--olive-dark);
}
.card-steel {
    background: var(--cool-steel);
    border-color: var(--steel-dark);
}
.card-bone {
    background: var(--bone);
    border-color: var(--bone-dark);
}

/* ── Info metric tiles ── */
.metric-tile {
    background: #fff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border: 1px solid var(--bone-dark);
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--olive);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-lbl {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-soft);
    font-weight: 500;
}

/* ── Slider label ── */
.input-label {
    font-family: 'Lora', serif;
    font-size: 0.88rem;
    font-style: italic;
    color: var(--ink-soft);
    margin-bottom: 0.25rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label,
div[data-testid="stNumberInput"] > label {
    font-family: 'Lora', serif !important;
    font-size: 0.9rem !important;
    font-style: italic !important;
    color: var(--ink-soft) !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 0.25rem;
}

/* ── Slider thumb color ── */
div[data-testid="stSlider"] [role="slider"] {
    background: var(--olive) !important;
    border-color: var(--olive) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: var(--bone-dark) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) {
    background: var(--olive) !important;
}

/* ── Radio buttons ── */
div[data-testid="stRadio"] [data-baseweb="radio"] {
    gap: 0.4rem !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    border-color: var(--bone-dark) !important;
    border-radius: 8px !important;
    background: #faf9f7 !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: var(--olive) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 2.2rem !important;
    transition: background 0.2s, transform 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--olive-dark) !important;
    transform: translateY(-1px) !important;
}

/* ── Result boxes ── */
.result-dropout {
    background: var(--warn-light);
    border: 2px solid var(--warn-red);
    border-radius: 12px;
    padding: 2rem 2.2rem;
    text-align: center;
}
.result-safe {
    background: #eef3f0;
    border: 2px solid #4a8060;
    border-radius: 12px;
    padding: 2rem 2.2rem;
    text-align: center;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    font-style: italic;
    margin-bottom: 0.4rem;
}
.result-dropout .result-title { color: var(--warn-red); }
.result-safe .result-title { color: #3a6b50; }
.result-prob {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0 0.3rem;
}
.result-dropout .result-prob { color: var(--warn-red); }
.result-safe .result-prob { color: #3a6b50; }
.result-desc {
    font-family: 'Lora', serif;
    font-style: italic;
    font-size: 0.92rem;
    color: var(--ink-soft);
    margin-top: 0.5rem;
}

/* ── Feature bar ── */
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.55rem;
}
.feat-name {
    font-family: 'Lora', serif;
    font-style: italic;
    font-size: 0.84rem;
    color: var(--ink-soft);
    width: 180px;
    flex-shrink: 0;
}
.feat-bar-wrap {
    flex: 1;
    background: var(--bone);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.feat-bar {
    height: 100%;
    border-radius: 4px;
}
.feat-val {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--ink-soft);
    width: 44px;
    text-align: right;
}

/* ── About section ── */
.about-card {
    background: var(--olive);
    border-radius: 12px;
    padding: 2rem 2.2rem;
    color: #fff;
}
.about-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 600;
    font-style: italic;
    color: var(--bone);
    margin-bottom: 0.8rem;
}
.about-card p, .about-card li {
    font-family: 'Lora', serif;
    font-size: 0.88rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.85);
}

/* ── Footer ── */
.footer {
    background: var(--ink);
    padding: 1.8rem 4rem;
    text-align: center;
    margin-top: 3rem;
}
.footer p {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.45);
    margin: 0;
    letter-spacing: 0.05em;
}

/* ── Responsive gap utilities ── */
.gap-sm { margin-bottom: 0.8rem; }
.gap-md { margin-bottom: 1.5rem; }
.gap-lg { margin-bottom: 2.5rem; }

/* ── Column spacer ── */
.col-spacer { padding: 0 0.5rem; }

/* ── Risk gauge text ── */
.gauge-wrap {
    position: relative;
    text-align: center;
    padding: 0.5rem 0;
}
.gauge-track {
    height: 14px;
    background: linear-gradient(90deg, #4a8060 0%, #d4a847 45%, var(--warn-red) 100%);
    border-radius: 8px;
    position: relative;
    margin: 0.8rem 0 0.4rem;
}
.gauge-needle {
    position: absolute;
    top: -5px;
    width: 4px;
    height: 24px;
    background: var(--ink);
    border-radius: 2px;
    transform: translateX(-50%);
    transition: left 0.5s ease;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--ink-soft);
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-label">Machine Learning · Educational Analytics</p>
    <h1 class="hero-title">Student Dropout<br>Predictor</h1>
    <p class="hero-sub">
        Identify students at risk of dropping out using logistic regression
        trained on academic and lifestyle indicators.
    </p>
    <span class="hero-badge">🎓 Logistic Regression · SMOTE · ROC-AUC 88%</span>
</div>
""", unsafe_allow_html=True)

# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Model Overview</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Performance metrics from test set evaluation</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metrics_data = [
    ("79.75%", "Accuracy"),
    ("83.06%", "Recall"),
    ("78.96%", "F1-Score"),
    ("88.29%", "ROC-AUC"),
]
for col, (val, lbl) in zip([m1, m2, m3, m4], metrics_data):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Student Profile Input</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Fill in the student\'s academic and lifestyle details below</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

left_col, spacer, right_col = st.columns([5, 0.4, 5])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**📚 Academic Indicators**")
    st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)

    gpa = st.slider(
        "GPA (Grade Point Average)",
        min_value=1.0, max_value=4.0, value=2.8, step=0.05,
        help="Student's current GPA on a 4.0 scale"
    )
    attendance = st.slider(
        "Attendance Rate (%)",
        min_value=0.0, max_value=100.0, value=75.0, step=0.5,
        help="Percentage of classes attended"
    )
    study_hours = st.slider(
        "Study Hours per Day",
        min_value=0.0, max_value=12.0, value=4.0, step=0.25,
        help="Average daily study hours"
    )
    assignment_delay = st.slider(
        "Assignment Delay (Days)",
        min_value=0, max_value=15, value=3,
        help="Average number of days assignments are submitted late"
    )
    semester = st.selectbox(
        "Current Semester / Year",
        options=["Year 1", "Year 2", "Year 3", "Year 4"],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🌱 Lifestyle & Support Indicators**")
    st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)

    stress = st.slider(
        "Stress Index (1 = Low, 10 = Very High)",
        min_value=1.0, max_value=10.0, value=5.0, step=0.1,
        help="Self-reported stress level"
    )
    travel_time = st.slider(
        "Travel Time to Campus (Minutes)",
        min_value=0, max_value=120, value=30,
        help="Daily commute time in minutes"
    )

    st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)
    internet = st.radio(
        "Has Internet Access?",
        options=["Yes", "No"],
        horizontal=True
    )
    part_time = st.radio(
        "Has Part-Time Job?",
        options=["Yes", "No"],
        horizontal=True
    )
    scholarship = st.radio(
        "Receiving Scholarship?",
        options=["Yes", "No"],
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    predict_clicked = st.button("✦ Predict Dropout Risk", use_container_width=True)

st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

# ── Result ────────────────────────────────────────────────────────────────────
if predict_clicked:
    input_df = pd.DataFrame({
        'Internet_Access': [internet],
        'Study_Hours_per_Day': [study_hours],
        'Attendance_Rate': [attendance],
        'Assignment_Delay_Days': [float(assignment_delay)],
        'Travel_Time_Minutes': [float(travel_time)],
        'Part_Time_Job': [part_time],
        'Scholarship': [scholarship],
        'Stress_Index': [stress],
        'GPA': [gpa],
        'Semester': [semester],
    })

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    dropout_prob = probabilities[1]
    safe_prob = probabilities[0]

    st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    res_left, res_right = st.columns([5, 5])

    with res_left:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-dropout">
                <div class="result-title">⚠ At Risk of Dropout</div>
                <div class="result-prob">{dropout_prob:.1%}</div>
                <div class="result-desc">Dropout probability — immediate intervention recommended</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-title">✓ Not at Risk</div>
                <div class="result-prob">{safe_prob:.1%}</div>
                <div class="result-desc">Safe probability — student appears on track</div>
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        needle_pos = int(dropout_prob * 100)
        st.markdown(f"""
        <div class="gauge-wrap" style="margin-top:1.2rem;">
            <div style="font-family:'Lora',serif; font-style:italic; font-size:0.82rem; color:var(--ink-soft); margin-bottom:0.3rem;">Risk Gauge</div>
            <div class="gauge-track">
                <div class="gauge-needle" style="left:{needle_pos}%;"></div>
            </div>
            <div class="gauge-labels"><span>Low Risk</span><span>Moderate</span><span>High Risk</span></div>
        </div>
        """, unsafe_allow_html=True)

    with res_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family:'Playfair Display',serif; font-size:1rem; font-weight:600; font-style:italic; margin-bottom:1rem; color:var(--ink);">
            Key Contributing Factors
        </p>
        """, unsafe_allow_html=True)

        # Normalized factor contributions
        factors = [
            ("GPA", max(0, 1 - gpa / 4), "#71713B"),
            ("Attendance", max(0, 1 - attendance / 100), "#71713B"),
            ("Stress Index", stress / 10, "#b85c42"),
            ("Study Hours", max(0, 1 - study_hours / 10), "#71713B"),
            ("Assignment Delay", assignment_delay / 15, "#b85c42"),
            ("Part-Time Job", 1.0 if part_time == "Yes" else 0.0, "#94A5B3"),
            ("No Internet", 1.0 if internet == "No" else 0.0, "#94A5B3"),
            ("No Scholarship", 1.0 if scholarship == "No" else 0.15, "#94A5B3"),
        ]

        for name, score, color in sorted(factors, key=lambda x: x[1], reverse=True):
            bar_w = int(score * 100)
            st.markdown(f"""
            <div class="feat-row">
                <span class="feat-name">{name}</span>
                <div class="feat-bar-wrap">
                    <div class="feat-bar" style="width:{bar_w}%; background:{color};"></div>
                </div>
                <span class="feat-val">{score:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

    # ── Recommendations ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Intervention Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    recs = []
    if attendance < 70:
        recs.append(("📅 Attendance Alert", "Attendance is critically low. Consider counseling and attendance monitoring programs."))
    if gpa < 2.5:
        recs.append(("📖 Academic Support", "GPA is below threshold. Tutoring and academic mentoring are strongly advised."))
    if stress > 7:
        recs.append(("🧘 Stress Management", "High stress levels detected. Refer to campus mental health services."))
    if study_hours < 2:
        recs.append(("⏰ Study Planning", "Very low study hours. Time management coaching may help."))
    if assignment_delay > 5:
        recs.append(("✏️ Assignment Tracking", "Frequent delays in assignment submission. A structured deadline tracker is recommended."))
    if part_time == "Yes" and scholarship == "No":
        recs.append(("💰 Financial Aid", "Student is working part-time without a scholarship. Explore financial aid opportunities."))
    if internet == "No":
        recs.append(("🌐 Digital Access", "No internet access may hinder learning. Explore campus connectivity programs."))

    if not recs:
        recs.append(("✓ No Critical Issues", "The student profile looks healthy. Continue regular monitoring."))

    rec_cols = st.columns(min(len(recs), 3))
    for i, (title, desc) in enumerate(recs):
        with rec_cols[i % 3]:
            bg = "var(--warn-light)" if prediction == 1 else "#eef3f0"
            border = "var(--warn-red)" if prediction == 1 else "#4a8060"
            st.markdown(f"""
            <div style="background:{bg}; border-left:3px solid {border}; border-radius:8px;
                        padding:1.1rem 1.2rem; margin-bottom:0.8rem;">
                <div style="font-family:'DM Sans',sans-serif; font-weight:500; font-size:0.88rem;
                            color:var(--ink); margin-bottom:0.35rem;">{title}</div>
                <div style="font-family:'Lora',serif; font-style:italic; font-size:0.82rem;
                            color:var(--ink-soft); line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

# ── About section ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">About This Model</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

ab1, ab2, ab3 = st.columns(3)

with ab1:
    st.markdown("""
    <div class="about-card">
        <h3>Algorithm</h3>
        <p>
            <strong>Logistic Regression</strong> with SMOTE oversampling to handle class imbalance.
            The model was selected based on superior Recall — critical for catching at-risk students
            before they drop out.
        </p>
    </div>
    """, unsafe_allow_html=True)

with ab2:
    st.markdown("""
    <div class="about-card">
        <h3>Features Used</h3>
        <ul>
            <li>Study Hours per Day</li>
            <li>Attendance Rate</li>
            <li>GPA</li>
            <li>Stress Index</li>
            <li>Assignment Delay Days</li>
            <li>Travel Time to Campus</li>
            <li>Internet Access, Part-Time Job, Scholarship, Semester</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with ab3:
    st.markdown("""
    <div class="about-card">
        <h3>Why Recall Matters</h3>
        <p>
            In dropout prediction, <em>missing an at-risk student</em> (False Negative) is
            far more costly than a false alarm. This model prioritises <strong>Recall</strong>
            to ensure maximum detection of students who need intervention.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrap

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>Student Dropout Predictor · Logistic Regression · Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
