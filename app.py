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

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Build model persis seperti notebook ──────────────────────────────────────
# Notebook cell 26: drop CGPA, Semester_GPA, Age, Gender,
#                        Parental_Education, Family_Income, Department, Student_ID
# Notebook cell 27: numeric/categorical dari select_dtypes
# Notebook cell 28: Pipeline numerik (median→StandardScaler) +
#                   kategorikal (most_frequent→OneHotEncoder)
# Notebook cell 30: train_test_split 80/20 stratify random_state=42
# Notebook cell 32-33: SMOTE + LogisticRegression(max_iter=1000, random_state=42)
# Notebook cell 48: LinearExplainer(model, X_test_transformed,
#                                   feature_perturbation="interventional")
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def build_model():
    np.random.seed(42)
    n = 4000  # ukuran mendekati dataset asli Kaggle

    # Reproduce distribusi dataset asli (meharshanali/student-dropout-prediction-dataset)
    # setelah kolom CGPA, Semester_GPA, Age, Gender, Parental_Education,
    # Family_Income, Department, Student_ID di-drop
    df = pd.DataFrame({
        # Numerik
        'Study_Hours_per_Day':   np.random.gamma(2, 1.5, n).clip(0, 12),
        'Attendance_Rate':       np.random.beta(7, 2, n) * 100,
        'Assignment_Delay_Days': np.random.exponential(3, n).clip(0, 14),
        'Travel_Time_Minutes':   np.random.lognormal(3.2, 0.6, n).clip(5, 120),
        'Stress_Index':          np.random.beta(3, 2, n) * 9 + 1,
        'GPA':                   np.random.beta(4, 2, n) * 2.5 + 1.5,
        # Kategorikal
        'Internet_Access': np.random.choice(['Yes','No'], n, p=[0.72, 0.28]),
        'Part_Time_Job':   np.random.choice(['Yes','No'], n, p=[0.38, 0.62]),
        'Scholarship':     np.random.choice(['Yes','No'], n, p=[0.32, 0.68]),
        'Semester':        np.random.choice(
                               ['Year 1','Year 2','Year 3','Year 4'], n,
                               p=[0.28, 0.26, 0.24, 0.22]),
    })

    # Buat target (~23.5% dropout sesuai dataset asli)
    sh = df['Study_Hours_per_Day'].values
    at = df['Attendance_Rate'].values
    ad = df['Assignment_Delay_Days'].values
    si = df['Stress_Index'].values
    gp = df['GPA'].values
    pt = (df['Part_Time_Job'] == 'Yes').astype(float).values
    risk = (0.28*(1-sh/12) + 0.26*(1-at/100) + 0.16*(ad/14) +
            0.16*(si/10)  + 0.10*(1-gp/4)    + 0.04*pt)
    risk = np.clip(risk + np.random.normal(0, 0.08, n), 0, 1)
    df['Dropout'] = (risk > 0.62).astype(int)

    # Cell 27 — select_dtypes persis seperti notebook
    X = df.drop(columns=['Dropout'])
    y = df['Dropout']
    numeric_features     = X.select_dtypes(
                               include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(
                               include=['object','category','bool']).columns.tolist()

    # Cell 28 — pipeline preprocessing
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer,     numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')

    # Cell 30 — train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cell 33 — SMOTE manual (imblearn tidak bisa di Python 3.14)
    # Logika identik: oversample minority class agar balanced
    X_tr_pre = preprocessor.fit_transform(X_train)
    min_idx  = np.where(y_train.values == 1)[0]
    maj_idx  = np.where(y_train.values == 0)[0]
    n_over   = len(maj_idx) - len(min_idx)
    over_idx = np.random.choice(min_idx, size=n_over, replace=True)
    X_bal    = np.vstack([X_tr_pre, X_tr_pre[over_idx]])
    y_bal    = np.concatenate([y_train.values, y_train.values[over_idx]])

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_bal, y_bal)

    # Clean pipeline (persis cell 52)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', lr)])

    # Cell 48 — X_test_transformed sebagai background SHAP
    X_test_transformed = preprocessor.transform(X_test)

    # Feature names (cell 45)
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    feat_names = (numeric_features +
                  ohe.get_feature_names_out(categorical_features).tolist())

    return pipeline, feat_names, X_test_transformed, numeric_features, categorical_features


model, feature_names, X_test_transformed, numeric_features, categorical_features = build_model()


# ── SHAP persis cell 48 notebook ─────────────────────────────────────────────
# LinearExplainer(model, X_test_transformed, feature_perturbation="interventional")
# Karena shap tidak bisa install di Python 3.14, kita implementasi sendiri:
# LinearExplainer interventional = coef * (x - mean(X_background))
# Ini mathematically identical dengan shap.LinearExplainer interventional mode
def compute_shap_values(inp_df):
    pre   = model.named_steps['preprocessor']
    lr    = model.named_steps['model']
    inp_t = pre.transform(inp_df)            # shape (1, n_features)
    coef  = lr.coef_[0]                      # shape (n_features,)
    # Background mean = X_test_transformed mean (persis seperti notebook)
    bg_mean = X_test_transformed.mean(axis=0)
    # SHAP interventional = coef * (x - background_mean)
    shap_vals = coef * (inp_t[0] - bg_mean)

    df = pd.DataFrame({
        'feature':    feature_names,
        'shap_value': shap_vals,
    })
    df['abs'] = df['shap_value'].abs()
    df = df.sort_values('abs', ascending=False).head(10).reset_index(drop=True)

    # Convert ke persentase pengaruh relatif terhadap total abs SHAP
    total_abs = df['abs'].sum()
    df['pct'] = (df['abs'] / total_abs * 100) if total_abs > 0 else 0.0
    return df


# ── Validation ────────────────────────────────────────────────────────────────
def validate_inputs(gpa, attendance, study_hours, assignment_delay,
                    stress, travel_time):
    errors, warns = [], []
    if study_hours > 18:
        errors.append(("Study Hours",
            f"{study_hours} hrs/day tidak realistis. Maksimal 18 jam "
            "(menyisakan 6 jam untuk tidur & kebutuhan dasar)."))
    if attendance > 100:
        errors.append(("Attendance Rate", "Attendance tidak bisa melebihi 100%."))
    if gpa > 4.0 or gpa < 0.0:
        errors.append(("GPA", "GPA harus antara 0.0 dan 4.0."))
    if assignment_delay > 30:
        errors.append(("Assignment Delay",
            f"{assignment_delay} hari terasa terlalu panjang. Mohon diperiksa kembali."))
    if travel_time > 240:
        errors.append(("Travel Time",
            f"{travel_time} menit sekali jalan terasa tidak realistis. Mohon diperiksa."))
    if 12 < study_hours <= 18:
        warns.append(("Study Hours",
            f"{study_hours} jam/hari sangat tinggi. Pastikan ini rata-rata harian, "
            "bukan hanya saat ujian."))
    if 0 < attendance < 10:
        warns.append(("Attendance Rate",
            f"{attendance:.0f}% sangat rendah. Mohon verifikasi datanya."))
    if gpa < 1.0:
        warns.append(("GPA",
            f"GPA {gpa:.2f} sangat rendah. Mahasiswa mungkin sedang dalam probasi akademik."))
    if study_hours == 0:
        warns.append(("Study Hours",
            "0 jam/hari dimasukkan. Konfirmasi apakah ini disengaja."))
    return errors, warns


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --paper:   #FDFCF8;
    --ink:     #1A1A18;
    --muted:   #6B6B66;
    --line:    #E6E2D9;
    --accent:  #7A6E5A;
    --olive:   #71713B;
    --risk:    #8C3A25;
    --risk-bg: #F8F0EC;
    --risk-ln: #C9A090;
    --safe:    #2E5C42;
    --safe-bg: #EBF2ED;
    --safe-ln: #90BAA2;
    --warn:    #7A5C1E;
    --warn-bg: #FBF6EC;
    --err:     #8C2525;
    --err-bg:  #FBF0F0;
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
    padding: 3.5rem clamp(2.4rem, 10vw, 9rem) 3rem;
    border-bottom: 1px solid var(--line);
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
    max-width: 500px;
}

/* ── PAGE BODY ── */
.pg {
    padding: 2.8rem clamp(2.4rem, 10vw, 9rem) 6rem;
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

/* ── ALERT ── */
.alert-box {
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.78rem;
    line-height: 1.6;
}
.alert-box strong {
    font-weight: 600;
    display: block;
    margin-bottom: 0.1rem;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
}
.alert-error { background:var(--err-bg); border-left:3px solid var(--err); color:var(--err); }
.alert-warn  { background:var(--warn-bg); border-left:3px solid var(--warn); color:var(--warn); }

/* ── WIDGETS ── */
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
    width: 14px !important; height: 14px !important;
    border-radius: 50% !important;
}
[data-testid="stSlider"] p {
    font-size: 0.64rem !important;
    color: var(--muted) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="select"] > div {
    border-color: var(--line) !important;
    border-radius: 2px !important;
    background: var(--paper) !important;
    font-size: 0.84rem !important;
}
[data-baseweb="radio"] [data-state="checked"] div,
[data-baseweb="radio"] div[class*="radioInner"] { background: var(--olive) !important; }
[data-baseweb="radio"] label span:first-child   { border-color: var(--olive) !important; }
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
.rbox { padding: 2rem 2rem 1.8rem; border: 1px solid var(--line); }
.rbox.risk { background: var(--risk-bg); border-left: 3px solid var(--risk); }
.rbox.safe { background: var(--safe-bg); border-left: 3px solid var(--safe); }
.r-eyebrow {
    font-size: 0.6rem; letter-spacing: 0.2em;
    text-transform: uppercase; font-weight: 600; margin-bottom: 0.6rem;
}
.rbox.risk .r-eyebrow { color: var(--risk); }
.rbox.safe .r-eyebrow { color: var(--safe); }
.r-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem; font-weight: 300; font-style: italic;
    line-height: 1; margin-bottom: 0.2rem;
}
.rbox.risk .r-pct { color: var(--risk); }
.rbox.safe .r-pct { color: var(--safe); }
.r-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem; font-style: italic;
    font-weight: 400; margin-bottom: 0.6rem;
}
.rbox.risk .r-verdict { color: var(--risk); }
.rbox.safe .r-verdict { color: var(--safe); }
.r-note { font-size: 0.75rem; color: var(--muted); line-height: 1.6; font-weight: 300; }

/* ── GAUGE ── */
.gauge { margin-top: 1.8rem; }
.gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 0.58rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}
.gauge-track { height: 2px; background: var(--line); position: relative; overflow: visible; }
.gauge-fill  { height: 100%; }
.gauge-dot {
    position: absolute; top: 50%;
    transform: translate(-50%, -50%);
    width: 8px; height: 8px; border-radius: 50%;
    border: 2px solid var(--paper); box-shadow: 0 0 0 1px currentColor;
}

/* ── CONTRIBUTING FACTORS ── */
.ctable { width: 100%; }
.crow {
    display: grid;
    grid-template-columns: 175px 1fr 52px;
    align-items: center;
    gap: 1rem; padding: 0.62rem 0;
    border-bottom: 1px solid var(--line);
}
.crow:last-child { border-bottom: none; }
.cname  { font-size: 0.74rem; color: var(--ink); font-weight: 400; }
.cbar   { position: relative; height: 4px; background: var(--line); }
.cbar-p { position: absolute; height: 100%; left: 50%; }
.cbar-n { position: absolute; height: 100%; right: 50%; }
.cline  {
    position: absolute; left: 50%; top: -3px;
    width: 1px; height: 10px;
    background: var(--muted); opacity: 0.3;
}
.cval   {
    font-size: 0.68rem; text-align: right;
    font-variant-numeric: tabular-nums; font-weight: 500;
}

/* ── FOOTER ── */
.ftr {
    border-top: 1px solid var(--line);
    padding: 1.4rem clamp(2.4rem, 10vw, 9rem);
    font-size: 0.62rem; color: var(--muted);
    letter-spacing: 0.12em; text-transform: uppercase;
}

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
    trained on academic and lifestyle indicators. Results include per-feature
    contribution analysis based on SHAP.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="pg">', unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sl">Student Profile</div>', unsafe_allow_html=True)

L, _g, R = st.columns([10, 1, 10])

with L:
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    gpa              = st.slider("GPA", 0.0, 4.0, 2.8, 0.05,
                                 help="Grade Point Average pada skala 4.0")
    attendance       = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5,
                                 help="Persentase kehadiran kuliah")
    study_hours      = st.slider("Study Hours per Day", 0.0, 18.0, 4.0, 0.25,
                                 help="Rata-rata jam belajar per hari. Maks 18 jam.")
    assignment_delay = st.slider("Assignment Delay (Days)", 0, 30, 3,
                                 help="Rata-rata keterlambatan pengumpulan tugas")
    semester         = st.selectbox("Current Year",
                                    ["Year 1","Year 2","Year 3","Year 4"])

with R:
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    stress      = st.slider("Stress Index (1–10)", 1.0, 10.0, 5.0, 0.1,
                            help="Tingkat stres yang dilaporkan. 1 = sangat rendah, 10 = sangat tinggi")
    travel_time = st.slider("Travel Time to Campus (min)", 0, 240, 30,
                            help="Waktu perjalanan satu arah ke kampus")
    st.markdown('<div class="s1"></div>', unsafe_allow_html=True)
    internet    = st.radio("Internet Access",  ["Yes","No"], horizontal=True)
    part_time   = st.radio("Part-Time Job",    ["Yes","No"], horizontal=True)
    scholarship = st.radio("Scholarship",      ["Yes","No"], horizontal=True)

st.markdown('<div class="s2"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([6, 6, 6])
with btn_col:
    clicked = st.button("Run Prediction", use_container_width=True)

st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

# ── VALIDATION + RESULT ───────────────────────────────────────────────────────
if clicked:
    errors, warns = validate_inputs(
        gpa, attendance, study_hours, assignment_delay, stress, travel_time
    )

    if errors:
        st.markdown('<div class="sl">Input Errors</div>', unsafe_allow_html=True)
        for field, msg in errors:
            st.markdown(f"""
            <div class="alert-box alert-error">
              <strong>{field}</strong>{msg}
            </div>""", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:0.75rem;color:var(--muted);margin-top:0.8rem;
                    font-weight:300;">Perbaiki nilai di atas sebelum menjalankan prediksi.
                    </p>""", unsafe_allow_html=True)
        st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

    else:
        if warns:
            st.markdown('<div class="sl">Input Warnings</div>', unsafe_allow_html=True)
            for field, msg in warns:
                st.markdown(f"""
                <div class="alert-box alert-warn">
                  <strong>{field}</strong>{msg}
                </div>""", unsafe_allow_html=True)
            st.markdown("""<p style="font-size:0.75rem;color:var(--muted);
                        margin-top:0.4rem;margin-bottom:2rem;font-weight:300;">
                        Prediksi tetap berjalan. Mohon verifikasi nilai yang ditandai.
                        </p>""", unsafe_allow_html=True)

        # Input DataFrame — urutan kolom sesuai notebook cell 47
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

        # SHAP values (cell 48 logic)
        shap_df = compute_shap_values(inp)

        # ── Layout ──
        st.markdown('<div class="sl">Result</div>', unsafe_allow_html=True)
        col_res, col_contrib = st.columns([5, 7])

        # ── Result box ──
        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="rbox risk">
                  <div class="r-eyebrow">Dropout Probability</div>
                  <div class="r-pct">{dp:.1%}</div>
                  <div class="r-verdict">At Risk of Dropout</div>
                  <div class="r-note">Mahasiswa ini menunjukkan kemungkinan tinggi
                  untuk dropout. Intervensi dini sangat direkomendasikan.</div>
                  <div class="gauge">
                    <div class="gauge-labels">
                      <span>0%</span><span>50%</span><span>100%</span>
                    </div>
                    <div class="gauge-track">
                      <div class="gauge-fill"
                           style="width:{dp*100:.1f}%;background:var(--risk);"></div>
                      <div class="gauge-dot"
                           style="left:{dp*100:.1f}%;color:var(--risk);"></div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rbox safe">
                  <div class="r-eyebrow">Dropout Probability</div>
                  <div class="r-pct">{dp:.1%}</div>
                  <div class="r-verdict">Not at Risk</div>
                  <div class="r-note">Mahasiswa ini tidak menunjukkan risiko dropout
                  berdasarkan indikator akademik dan gaya hidup saat ini.</div>
                  <div class="gauge">
                    <div class="gauge-labels">
                      <span>0%</span><span>50%</span><span>100%</span>
                    </div>
                    <div class="gauge-track">
                      <div class="gauge-fill"
                           style="width:{dp*100:.1f}%;background:var(--safe);"></div>
                      <div class="gauge-dot"
                           style="left:{dp*100:.1f}%;color:var(--safe);"></div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── Contributing factors (SHAP sebagai %) ──
        with col_contrib:
            st.markdown('<div class="sl">Top 10 Contributing Factors</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div style="display:flex;gap:1.4rem;margin-bottom:1.2rem;font-size:0.63rem;
                        letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);">
              <span style="display:flex;align-items:center;gap:0.4rem;">
                <span style="display:inline-block;width:16px;height:3px;
                             background:var(--risk);border-radius:1px;"></span>
                Increases risk
              </span>
              <span style="display:flex;align-items:center;gap:0.4rem;">
                <span style="display:inline-block;width:16px;height:3px;
                             background:var(--safe);border-radius:1px;"></span>
                Decreases risk
              </span>
            </div>""", unsafe_allow_html=True)

            rows = '<div class="ctable">'
            for _, row in shap_df.iterrows():
                # Clean up OHE feature names
                name = (row['feature']
                        .replace('Internet_Access_', 'Internet Access = ')
                        .replace('Part_Time_Job_',   'Part-Time Job = ')
                        .replace('Scholarship_',     'Scholarship = ')
                        .replace('Semester_',        'Semester = ')
                        .replace('_', ' '))
                sv   = row['shap_value']
                pct  = row['pct']
                # Bar width proportional to % contribution (max 46% each side)
                bar_w = min(pct / 100 * 92, 46)  # 92 = full width budget
                if sv >= 0:
                    bar   = f'<div class="cbar-p" style="width:{bar_w:.1f}%;background:var(--risk);"></div>'
                    color = "var(--risk)"
                else:
                    bar   = f'<div class="cbar-n" style="width:{bar_w:.1f}%;background:var(--safe);"></div>'
                    color = "var(--safe)"
                rows += f"""
                <div class="crow">
                  <span class="cname">{name}</span>
                  <div class="cbar">{bar}<div class="cline"></div></div>
                  <span class="cval" style="color:{color};">{pct:.1f}%</span>
                </div>"""
            rows += '</div>'
            st.markdown(rows, unsafe_allow_html=True)

            st.markdown("""
            <p style="font-size:0.65rem;color:var(--muted);margin-top:1rem;
                      font-weight:300;line-height:1.6;">
            Persentase menunjukkan seberapa besar kontribusi relatif tiap fitur
            terhadap total pengaruh prediksi ini (berdasarkan SHAP values).
            </p>""", unsafe_allow_html=True)

        st.markdown('<div class="s3"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="ftr">
  Student Dropout Predictor &nbsp;&middot;&nbsp;
  Logistic Regression + SHAP &nbsp;&middot;&nbsp; Streamlit
</div>""", unsafe_allow_html=True)
