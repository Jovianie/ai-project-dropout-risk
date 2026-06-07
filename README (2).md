# 🎓 Student Dropout Predictor

A minimal, aesthetic Streamlit web application that predicts student dropout risk using a **Logistic Regression** model trained with SMOTE oversampling.

---

## ✦ Live Demo

Deploy this app to Streamlit Cloud in minutes — see the [Deployment](#deployment) section below.

---

## Features

- **Dropout Probability Prediction** — real-time prediction with dropout probability percentage
- **Risk Gauge** — visual indicator ranging from low to high risk
- **Key Contributing Factors** — bar chart of the most influential features for each prediction
- **Intervention Recommendations** — contextual, actionable advice tailored to the student's profile
- **Model Overview Panel** — accuracy, recall, F1-score, and ROC-AUC displayed upfront

---

## Model Details

| Detail | Info |
|---|---|
| Algorithm | Logistic Regression |
| Oversampling | SMOTE (to handle class imbalance) |
| Accuracy | 79.75% |
| Recall | 83.06% |
| F1-Score | 78.96% |
| ROC-AUC | 88.29% |

### Features Used

| Feature | Type |
|---|---|
| Study Hours per Day | Numeric |
| Attendance Rate (%) | Numeric |
| GPA | Numeric |
| Stress Index (1–10) | Numeric |
| Assignment Delay Days | Numeric |
| Travel Time to Campus (min) | Numeric |
| Internet Access | Categorical |
| Part-Time Job | Categorical |
| Scholarship | Categorical |
| Semester / Year | Categorical |

> **Why Recall?** In dropout prediction, missing an at-risk student (False Negative) is more costly than a false alarm. The model prioritises Recall to ensure maximum detection.

---

## Project Structure

```
student-dropout-app/
├── app.py                          # Main Streamlit application
├── logistic_regression_model.pkl   # Trained model (pipeline with preprocessor + SMOTE + LR)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd student-dropout-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push this repository to GitHub (make sure `logistic_regression_model.pkl` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> ⚠️ Make sure `logistic_regression_model.pkl` is committed to your repository. The app loads it at startup.

---

## Generating the Model (Optional)

If you want to retrain the model from scratch using the original notebook:

1. Open `prediksi_student_dropout.ipynb` in Jupyter or Google Colab
2. Run all cells
3. The final cell exports `logistic_regression_model.pkl` via:
   ```python
   import joblib
   joblib.dump(best_model, "logistic_regression_model.pkl")
   ```
4. Move the `.pkl` file to this project folder

---

## Design

Color palette inspired by a **Cool Steel × Dusty Olive × Bone** moodboard:

| Name | Hex |
|---|---|
| Cool Steel | `#94A5B3` |
| Dusty Olive | `#71713B` |
| Bone | `#E2DCD0` |

Typography: **Playfair Display** (display/headers) · **Lora** (body italic) · **DM Sans** (UI labels)

---

## License

MIT — free to use, modify, and distribute.
