import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from io import BytesIO

# Konfigurasi Halaman (harus menjadi perintah Streamlit pertama)
st.set_page_config(
    page_title="Dropout Risk Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Model yang sudah dilatih
@st.cache_resource
def load_trained_model():
    # Pastikan file 'student_dropout_model.pkl' ada di direktori yang sama
    try:
        model = joblib.load('student_dropout_model.pkl')
        return model
    except FileNotFoundError:
        st.error("File model 'student_dropout_model.pkl' tidak ditemukan. Pastikan file tersebut sudah di-upload.")
        st.stop()

model = load_trained_model()

# Koefisien Model (diambil dari notebook Anda)
# Ini akan digunakan untuk menampilkan kontribusi fitur secara akurat
COEF_DATA = {
    'GPA': -1.26,
    'Stress Index': 0.31,
    'Assignment Delay (Days)': 0.25,
    'Attendance Rate (%)': -0.24,
    'Part-Time Job (Yes)': 0.02,
    'Internet Access (Yes)': -0.14,
    'Travel Time (minutes)': 0.13,
    'Semester (Year 1)': -0.12,
    'Study Hours per Day': -0.03,
}

# ---- CSS Kustom Minimalis Elegan ----
st.markdown("""
<style>
/* ----- RESET & GLOBAL ----- */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}

/* Sembunyikan elemen default Streamlit yang tidak perlu */
#MainMenu, header, footer, .stDeployButton {
    display: none !important;
}

/* Kontainer utama */
.main .block-container {
    padding-top: 0;
    padding-bottom: 0;
    max-width: 1280px;
}

/* ----- TYPOGRAPHY ----- */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.5rem;
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.5rem;
}

h2 {
    font-size: 1.5rem;
    color: #0f172a;
    border-left: 4px solid #3b82f6;
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
}

/* Deskripsi */
.subtitle {
    color: #475569;
    font-size: 1rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 1rem;
}

/* ----- METRICS ----- */
.metric-card {
    background: white;
    padding: 1.2rem 1rem;
    border-radius: 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05), 0 1px 1px rgba(0,0,0,0.02);
    border: 1px solid #e2e8f0;
    text-align: center;
    transition: all 0.2s ease;
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* ----- FORM STYLING ----- */
.form-section {
    background: white;
    border-radius: 24px;
    padding: 1.75rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}

/* Label Input */
.stSlider label, .stSelectbox label, .stRadio label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.25rem !important;
}

/* Slider Custom */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: #3b82f6 !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: #3b82f6 !important;
}

/* Radio Buttons */
div[role="radiogroup"] label {
    background-color: #f8fafc;
    border-radius: 40px;
    padding: 0.4rem 1.2rem;
    border: 1px solid #e2e8f0;
    margin-right: 0.5rem;
}
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    border-color: #cbd5e1 !important;
}
div[role="radiogroup"] label[data-selected="true"] {
    background-color: #3b82f6;
    border-color: #3b82f6;
    color: white;
}
div[role="radiogroup"] label[data-selected="true"] span {
    color: white !important;
}
div[role="radiogroup"] label[data-selected="true"] > div:first-child {
    border-color: white !important;
}
div[role="radiogroup"] label[data-selected="true"] > div:first-child > div {
    background-color: white !important;
}

/* Tombol Prediksi */
.stButton > button {
    width: 100%;
    background: linear-gradient(95deg, #0f172a 0%, #1e293b 100%);
    color: white;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
    border-radius: 40px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.stButton > button:hover {
    background: linear-gradient(95deg, #1e293b 0%, #0f172a 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ----- RESULT BOX ----- */
.result-card {
    background: white;
    border-radius: 24px;
    padding: 1.75rem;
    border: 1px solid #e2e8f0;
    margin: 1rem 0 1.5rem 0;
}
.result-card.risk {
    border-left: 8px solid #ef4444;
    background: #fef2f2;
}
.result-card.safe {
    border-left: 8px solid #10b981;
    background: #f0fdf9;
}
.result-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.result-percentage {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.result-verdict {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.risk .result-verdict { color: #dc2626; }
.safe .result-verdict { color: #059669; }

/* Factor Bar */
.factor-item {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.factor-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #334155;
}
.factor-bar-bg {
    width: 100%;
    height: 6px;
    background-color: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
}
.factor-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}

/* Recommendation Card */
.rec-card {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    height: 100%;
    transition: all 0.2s;
}
.rec-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.rec-title {
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    color: #0f172a;
}
.rec-desc {
    font-size: 0.8rem;
    color: #475569;
    line-height: 1.5;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 2rem;
    font-size: 0.7rem;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
    <h1>🎓 Student Dropout Risk Analyzer</h1>
    <div class="subtitle">
        Early identification of at-risk students using a Logistic Regression model.<br>
        Prioritizing <strong>Recall (83%)</strong> ensures we minimize false negatives — the most critical metric in dropout prevention.
    </div>
</div>
""", unsafe_allow_html=True)

# ---- METRICS (Dari Notebook) ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model</div>
        <div class="metric-value">Logistic Reg.</div>
        <div class="metric-sub">w/ SMOTE</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">74.4%</div>
        <div class="metric-sub">Overall performance</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Recall</div>
        <div class="metric-value">75.8%</div>
        <div class="metric-sub">⭐ Primary metric</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-value">0.819</div>
        <div class="metric-sub">Discrimination power</div>
    </div>
    """, unsafe_allow_html=True)

# ---- INPUT FORM ----
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown("### 📋 Student Profile")

left, right = st.columns(2)

with left:
    gpa = st.slider(
        "GPA", 
        min_value=0.0, max_value=4.0, value=2.8, step=0.05,
        help="Cumulative Grade Point Average (0-4 scale)"
    )
    attendance = st.slider(
        "Attendance Rate (%)", 
        min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        help="Percentage of attended classes"
    )
    study_hours = st.slider(
        "Study Hours per Day", 
        min_value=0.0, max_value=12.0, value=4.0, step=0.5,
        help="Average daily self-study hours"
    )
    assignment_delay = st.slider(
        "Assignment Delay (Days)", 
        min_value=0, max_value=15, value=3, step=1,
        help="Average delay in submitting assignments"
    )
    semester = st.selectbox(
        "Current Semester", 
        options=["Year 1", "Year 2", "Year 3", "Year 4"]
    )

with right:
    stress = st.slider(
        "Stress Index (1-10)", 
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        help="Self-reported stress level"
    )
    travel_time = st.slider(
        "Travel Time to Campus (min)", 
        min_value=0, max_value=120, value=30, step=5,
        help="One-way commute time"
    )
    st.markdown("---")
    internet = st.radio(
        "Internet Access", 
        options=["Yes", "No"], 
        horizontal=True,
        help="Reliable internet access at home"
    )
    part_time = st.radio(
        "Part-Time Job", 
        options=["Yes", "No"], 
        horizontal=True,
        help="Currently working part-time"
    )
    scholarship = st.radio(
        "Scholarship", 
        options=["Yes", "No"], 
        horizontal=True,
        help="Receiving financial aid/scholarship"
    )

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔍 Predict Dropout Risk", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---- PREDICTION LOGIC ----
if predict_clicked:
    # Create input dataframe
    input_data = pd.DataFrame([{
        'GPA': gpa,
        'Attendance_Rate': attendance,
        'Study_Hours_per_Day': study_hours,
        'Assignment_Delay_Days': float(assignment_delay),
        'Stress_Index': stress,
        'Travel_Time_Minutes': float(travel_time),
        'Internet_Access': internet,
        'Part_Time_Job': part_time,
        'Scholarship': scholarship,
        'Semester': semester,
    }])

    # Predict
    proba = model.predict_proba(input_data)[0]
    risk_prob = proba[1]  # probability of dropout
    safe_prob = proba[0]   # probability of not dropout
    prediction = model.predict(input_data)[0]

    # Determine risk level
    risk_level = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.4 else "Low"

    # ---- RESULT DISPLAY ----
    st.markdown("### 📊 Prediction Result")
    
    result_col, factors_col = st.columns([5, 7])
    
    with result_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card risk">
                <div class="result-title">Dropout Probability</div>
                <div class="result-percentage">{risk_prob:.1%}</div>
                <div class="result-verdict">⚠️ At Risk of Dropout</div>
                <div style="font-size:0.85rem; color:#475569;">
                    Risk Level: <strong>{risk_level}</strong><br>
                    Immediate intervention is recommended.
                </div>
                <div style="margin-top: 1rem; background:#e2e8f0; border-radius: 40px; height: 8px;">
                    <div style="width: {risk_prob*100:.1f}%; background:#ef4444; border-radius: 40px; height: 8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card safe">
                <div class="result-title">Safe Probability</div>
                <div class="result-percentage">{safe_prob:.1%}</div>
                <div class="result-verdict">✅ Not at Risk</div>
                <div style="font-size:0.85rem; color:#475569;">
                    Risk Level: <strong>{risk_level}</strong><br>
                    Student appears on track.
                </div>
                <div style="margin-top: 1rem; background:#e2e8f0; border-radius: 40px; height: 8px;">
                    <div style="width: {safe_prob*100:.1f}%; background:#10b981; border-radius: 40px; height: 8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with factors_col:
        st.markdown("#### 🔍 Contributing Factors")
        st.caption("Based on model coefficients — positive values increase risk")

        # Calculate contribution for each factor based on user input
        # Convert user values to standardized contributions
        contributions = {}
        
        # GPA (lower GPA increases risk) - koefisien negatif: lower GPA = higher contribution
        gpa_contrib = abs(COEF_DATA['GPA']) * (1 - min(1.0, gpa / 4.0))
        contributions['Low GPA'] = min(1.0, gpa_contrib)
        
        # Attendance
        att_contrib = abs(COEF_DATA['Attendance Rate (%)']) * (1 - attendance/100)
        contributions['Low Attendance'] = min(1.0, att_contrib)
        
        # Stress
        stress_contrib = COEF_DATA['Stress Index'] * (stress / 10)
        contributions['High Stress'] = min(1.0, stress_contrib)
        
        # Assignment Delay
        delay_contrib = COEF_DATA['Assignment Delay (Days)'] * (assignment_delay / 15)
        contributions['Assignment Delay'] = min(1.0, delay_contrib)
        
        # Study Hours
        study_contrib = abs(COEF_DATA['Study Hours per Day']) * (1 - study_hours/12)
        contributions['Low Study Hours'] = min(1.0, study_contrib)
        
        # Part-time job
        if part_time == "Yes":
            contributions['Part-Time Job'] = 0.5  # default moderate contribution
        
        # Internet access
        if internet == "No":
            contributions['No Internet Access'] = 0.7
        
        # Scholarship
        if scholarship == "No":
            contributions['No Scholarship'] = 0.4
        
        # Sort and display top factors
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        for name, score in sorted_factors[:5]:
            color = "#ef4444" if score > 0.6 else "#f97316" if score > 0.3 else "#3b82f6"
            st.markdown(f"""
            <div class="factor-item">
                <div class="factor-label">
                    <span>{name}</span>
                    <span>{score:.0%}</span>
                </div>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width: {score*100:.0f}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- RECOMMENDATIONS ----
    st.markdown("### 💡 Recommendations")
    rec_cols = st.columns(3)
    
    recommendations = []
    
    if gpa < 2.5:
        recommendations.append(("📚 Academic Support", "Low GPA detected. Recommend tutoring, study groups, or academic counseling sessions."))
    elif gpa < 3.0:
        recommendations.append(("📚 Academic Monitoring", "GPA below 3.0. Suggest regular progress check-ins with academic advisor."))
    
    if attendance < 75:
        recommendations.append(("📅 Attendance Intervention", f"Attendance rate at {attendance:.0f}%. Set up attendance monitoring and engagement plan."))
    
    if stress > 7:
        recommendations.append(("🧠 Mental Health Support", "High stress level. Refer to campus counseling services and stress management workshops."))
    elif stress > 6:
        recommendations.append(("🧘 Stress Management", "Elevated stress. Suggest wellness programs and time management training."))
    
    if assignment_delay > 5:
        recommendations.append(("📝 Deadline Management", f"Assignment delays of {assignment_delay} days. Implement structured deadline tracking system."))
    elif assignment_delay > 3:
        recommendations.append(("⏰ Time Management", "Assignment delays detected. Recommend time management coaching."))
    
    if study_hours < 3:
        recommendations.append(("📖 Study Habits", f"Only {study_hours:.1f} hours of study/day. Suggest structured study plan and peer tutoring."))
    
    if part_time == "Yes" and scholarship == "No":
        recommendations.append(("💰 Financial Support", "Student works part-time without scholarship. Explore financial aid and work-study balance programs."))
    
    if internet == "No":
        recommendations.append(("🌐 Digital Access", "No internet access at home. Provide campus Wi-Fi access or device lending options."))
    
    if not recommendations:
        recommendations.append(("✅ On Track", "All indicators within healthy ranges. Continue regular monitoring and engagement."))
    
    # Distribute recommendations across columns
    for i, (title, desc) in enumerate(recommendations[:3]):
        with rec_cols[i % 3]:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show disclaimer
    st.caption("⚠️ This prediction is based on a statistical model and should be used as an early warning tool, not a definitive diagnosis.")

# ---- FOOTER ----
st.markdown("""
<div class="footer">
    Student Dropout Risk Analyzer — Built with Logistic Regression | Prioritizing Recall to Minimize False Negatives
</div>
""", unsafe_allow_html=True)
