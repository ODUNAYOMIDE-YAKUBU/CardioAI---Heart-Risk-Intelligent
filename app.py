import sys
import sklearn.compose._column_transformer

# Fix for sklearn version mismatch
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# ================================
# 📦 IMPORTS
# ================================
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ================================
# 🎨 PAGE CONFIG
# ================================
st.set_page_config(
    page_title="CardioAI · Heart Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# 🎨 GLOBAL CSS — Medical Dark Theme
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-base:       #080c14;
    --bg-card:       #0d1320;
    --bg-card2:      #111827;
    --bg-input:      #141d2e;
    --border:        #1e2d45;
    --border-glow:   #1e4976;
    --accent-blue:   #2f9cf7;
    --accent-teal:   #0ecfc4;
    --accent-red:    #ff4d6d;
    --accent-amber:  #f7b731;
    --accent-green:  #2dd4a0;
    --text-primary:  #e8f0fe;
    --text-secondary:#7b90b2;
    --text-muted:    #4a5f7a;
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'Space Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-base) !important; }
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1400px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero header ── */
.hero-wrap {
    background: linear-gradient(135deg, #090f1e 0%, #0a1628 50%, #061020 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(47,156,247,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(14,207,196,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: var(--font-display) !important;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #e8f0fe 0%, #2f9cf7 60%, #0ecfc4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(47,156,247,0.1);
    border: 1px solid rgba(47,156,247,0.3);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-family: var(--font-mono) !important;
    color: var(--accent-blue);
    letter-spacing: 1px;
    margin-top: 1rem;
}
.pulse-dot {
    width: 7px; height: 7px;
    background: var(--accent-teal);
    border-radius: 50%;
    animation: pulse 2s infinite;
    display: inline-block;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(1.4); }
}

/* ── Section labels ── */
.section-label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-glow), transparent);
}

/* ── Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    position: relative;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--border-glow); }
.metric-card-accent { border-left: 3px solid var(--accent-blue); }

/* ── Risk meter ── */
.risk-display {
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.2rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.risk-high {
    background: linear-gradient(135deg, rgba(255,77,109,0.12), rgba(255,77,109,0.04));
    border-color: rgba(255,77,109,0.35);
}
.risk-moderate {
    background: linear-gradient(135deg, rgba(247,183,49,0.12), rgba(247,183,49,0.04));
    border-color: rgba(247,183,49,0.35);
}
.risk-low {
    background: linear-gradient(135deg, rgba(45,212,160,0.12), rgba(45,212,160,0.04));
    border-color: rgba(45,212,160,0.35);
}
.risk-pct {
    font-family: var(--font-display) !important;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}
.risk-high   .risk-pct { color: var(--accent-red); }
.risk-moderate .risk-pct { color: var(--accent-amber); }
.risk-low    .risk-pct { color: var(--accent-green); }
.risk-label {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
    opacity: 0.75;
}
.risk-bar-track {
    background: rgba(255,255,255,0.07);
    border-radius: 50px;
    height: 6px;
    margin-top: 1.2rem;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}

/* ── Inputs ── */
.stSlider > div > div { background: var(--border) !important; }
.stSlider > div > div > div { background: var(--accent-blue) !important; }

div[data-baseweb="select"] > div {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(47,156,247,0.15) !important;
}

input[type="number"] {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

.stNumberInput > div > div {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a6fc4, #0ecfc4) !important;
    color: white !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 20px rgba(14,207,196,0.2) !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(14,207,196,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Info box ── */
.interp-box {
    background: linear-gradient(135deg, rgba(47,156,247,0.07), rgba(14,207,196,0.05));
    border: 1px solid rgba(47,156,247,0.2);
    border-left: 4px solid var(--accent-blue);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.93rem;
    line-height: 1.75;
    margin-top: 1.2rem;
    color: var(--text-primary);
}
.interp-title {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--accent-teal);
    margin-bottom: 0.7rem;
    display: block;
}

/* ── Table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stDataFrame thead tr th {
    background: var(--bg-card2) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
.stDataFrame tbody tr td {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
    border-color: var(--border) !important;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
    margin: 1.8rem 0;
}

/* ── Input group label ── */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-family: var(--font-body) !important;
    font-size: 0.83rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

/* ── Matplotlib figure background ── */
.stPlotlyChart, .stPyplot { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ================================
# 🏷️ CLEAN FEATURE LABEL MAP
# ================================
FEATURE_LABEL_MAP = {
    "num__age":                                    "Age (years)",
    "num__resting_bp":                             "Resting Blood Pressure (mmHg)",
    "num__serum_chol":                             "Serum Cholesterol (mg/dL)",
    "num__max_heart_rate":                         "Max Heart Rate Achieved (bpm)",
    "num__oldpeak":                                "ST Depression (Oldpeak)",
    "num__no_vessel":                              "Major Vessels (Fluoroscopy)",
    "cat__sex_male":                               "Sex: Male",
    "cat__sex_female":                             "Sex: Female",
    "cat__chest_pain_typical angina":              "Chest Pain: Typical Angina",
    "cat__chest_pain_atypical angina":             "Chest Pain: Atypical Angina",
    "cat__chest_pain_non-anginal":                 "Chest Pain: Non-Anginal",
    "cat__chest_pain_asymptomatic":                "Chest Pain: Asymptomatic",
    "cat__fasting_blood_sugar_False":              "Fasting Blood Sugar ≤ 120 mg/dL",
    "cat__fasting_blood_sugar_True":               "Fasting Blood Sugar > 120 mg/dL",
    "cat__resting_electrocardiog_normal":          "Resting ECG: Normal",
    "cat__resting_electrocardiog_lv hypertrophy":  "Resting ECG: LV Hypertrophy",
    "cat__resting_electrocardiog_st-t abnormality":"Resting ECG: ST-T Abnormality",
    "cat__exang_True":                             "Exercise-Induced Angina: Yes",
    "cat__exang_False":                            "Exercise-Induced Angina: No",
    "cat__slope_upsloping":                        "ST Slope: Upsloping",
    "cat__slope_flat":                             "ST Slope: Flat",
    "cat__slope_downsloping":                      "ST Slope: Downsloping",
    "cat__thal_normal":                            "Thalassemia: Normal",
    "cat__thal_fixed defect":                      "Thalassemia: Fixed Defect",
    "cat__thal_reversable defect":                 "Thalassemia: Reversible Defect",
}

def clean_label(raw):
    return FEATURE_LABEL_MAP.get(raw,
        raw.replace("cat__", "").replace("num__", "").replace("_", " ").title())


# ================================
# 📥 LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_heart_model.joblib")

pipeline = load_model()


# ================================
# 🏠 HERO HEADER
# ================================
st.markdown("""
<div class="hero-wrap">
    <p class="hero-sub">🫀 Explainable AI · Clinical Decision Support</p>
    <h1 class="hero-title">CardioAI<br>Risk Intelligence</h1>
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        XGBoost + SHAP · AUC 0.92 · 88% Accuracy
    </div>
</div>
""", unsafe_allow_html=True)


# ================================
# 🧍 INPUT  |  📊 RESULTS  LAYOUT
# ================================
left_col, right_col = st.columns([1, 1.55], gap="large")

# ────────────────────────────────
# LEFT PANEL — Patient Input Form
# ────────────────────────────────
with left_col:
    st.markdown('<div class="section-label">01 · Patient Profile</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 20, 80, 52, help="Patient age in years")
        with c2:
            sex = st.selectbox("Biological Sex", ["Male", "Female"])

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · Cardiac Symptoms</div>', unsafe_allow_html=True)

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"],
        help="Type of chest pain reported by patient"
    )
    exang = st.selectbox(
        "Exercise-Induced Angina",
        [False, True],
        format_func=lambda x: "Yes — angina during exercise" if x else "No — no angina during exercise"
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · Haemodynamic Measurements</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        resting_bp = st.number_input("Resting BP (mmHg)", 80, 200, 130)
    with c4:
        max_hr = st.number_input("Max Heart Rate (bpm)", 60, 220, 152)

    c5, c6 = st.columns(2)
    with c5:
        serum_chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 212)
    with c6:
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · Diagnostic Tests</div>', unsafe_allow_html=True)

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [False, True],
        format_func=lambda x: "> 120 mg/dL  (High)" if x else "≤ 120 mg/dL  (Normal)"
    )
    rest_ecg = st.selectbox(
        "Resting ECG Result",
        ["Normal", "LV Hypertrophy", "ST-T Abnormality"],
        help="Results of the resting electrocardiogram"
    )

    c7, c8 = st.columns(2)
    with c7:
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    with c8:
        no_vessel = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3],
                                  help="Number of major vessels coloured by fluoroscopy")

    thal = st.selectbox(
        "Thalassemia Type",
        ["Normal", "Fixed Defect", "Reversible Defect"],
        help="Thalassemia classification from nuclear stress test"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("🔍  Analyse Patient Risk", use_container_width=True)


# ────────────────────────────────
# RIGHT PANEL — Results
# ────────────────────────────────
with right_col:
    st.markdown('<div class="section-label">05 · AI Diagnosis & Explanation</div>', unsafe_allow_html=True)

    if not analyse:
        st.markdown("""
        <div style="
            background: var(--bg-card);
            border: 1px dashed var(--border);
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            color: var(--text-muted);
        ">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🫀</div>
            <div style="font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem; color: var(--text-secondary);">
                Awaiting Patient Data
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; max-width: 300px; margin: 0 auto;">
                Complete the patient profile and press <strong style="color: var(--accent-blue)">Analyse Patient Risk</strong> to generate the AI prediction and SHAP explanation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Build input DataFrame ──
        input_data = pd.DataFrame([{
            "age":                    age,
            "sex":                    sex.lower(),
            "chest_pain":             chest_pain.lower(),
            "resting_bp":             resting_bp,
            "serum_chol":             serum_chol,
            "fasting_blood_sugar":    fasting_bs,
            "resting_electrocardiog": rest_ecg.lower(),
            "max_heart_rate":         max_hr,
            "exang":                  exang,
            "oldpeak":                oldpeak,
            "slope":                  slope.lower(),
            "no_vessel":              no_vessel,
            "thal":                   thal.lower(),
        }])

        try:
            # ── Prediction ──
            prediction  = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]
            pct         = probability * 100

            # ── Risk classification ──
            if pct >= 70:
                risk_class = "risk-high"
                risk_icon  = "🔴"
                risk_text  = "HIGH RISK"
                bar_color  = "#ff4d6d"
            elif pct >= 30:
                risk_class = "risk-moderate"
                risk_icon  = "🟡"
                risk_text  = "MODERATE RISK"
                bar_color  = "#f7b731"
            else:
                risk_class = "risk-low"
                risk_icon  = "🟢"
                risk_text  = "LOW RISK"
                bar_color  = "#2dd4a0"

            # ── Risk display card ──
            st.markdown(f"""
            <div class="risk-display {risk_class}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span style="font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:3px;
                                     text-transform:uppercase; color:var(--text-muted);">
                            Predicted Heart Disease Probability
                        </span>
                        <p class="risk-pct">{pct:.1f}%</p>
                        <p class="risk-label">{risk_icon} {risk_text}</p>
                    </div>
                    <div style="text-align:right; font-family:'Space Mono',monospace;
                                font-size:0.72rem; color:var(--text-muted); margin-top:4px;">
                        <div>AUC-ROC · 0.9207</div>
                        <div style="margin-top:4px;">XGBoost Model</div>
                        <div style="margin-top:4px; color:var(--accent-teal);">SHAP Explained ✓</div>
                    </div>
                </div>
                <div class="risk-bar-track">
                    <div class="risk-bar-fill"
                         style="width:{min(pct, 100):.1f}%; background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── SHAP computation ──
            model        = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessing"]
            transformed  = preprocessor.transform(input_data)
            raw_names    = preprocessor.get_feature_names_out()
            clean_labels = [clean_label(n) for n in raw_names]

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)

            if isinstance(shap_values, list):
                sv   = shap_values[1][0]
                base = explainer.expected_value[1] if isinstance(
                    explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv   = shap_values[0] if shap_values.ndim > 1 else shap_values
                base = explainer.expected_value

            data_row = transformed[0] if transformed.ndim > 1 else transformed

            shap_df = pd.DataFrame({
                "Feature": clean_labels,
                "Impact":  sv
            }).sort_values("Impact", key=abs, ascending=True)

            # ── BAR CHART ──
            st.markdown('<div class="section-label" style="margin-top:1.4rem;">AI Explanation · Feature Contributions</div>',
                        unsafe_allow_html=True)

            # Show top 14 most impactful features
            plot_df = shap_df.copy()
            if len(plot_df) > 14:
                others_sum = plot_df.iloc[:-14]["Impact"].sum()
                plot_df = plot_df.tail(14)
                other_row = pd.DataFrame([{"Feature": f"({len(shap_df)-14} other features)", "Impact": others_sum}])
                plot_df = pd.concat([other_row, plot_df], ignore_index=True)

            n_rows     = len(plot_df)
            bar_height = max(5.5, n_rows * 0.42)

            fig1, ax1 = plt.subplots(figsize=(9, bar_height))
            fig1.patch.set_facecolor("#0d1320")
            ax1.set_facecolor("#0d1320")

            colors = ["#ff4d6d" if x > 0 else "#2f9cf7" for x in plot_df["Impact"]]
            bars   = ax1.barh(plot_df["Feature"], plot_df["Impact"],
                              color=colors, edgecolor="none", height=0.62)

            # Value labels
            for bar, val in zip(bars, plot_df["Impact"]):
                pad = max(abs(plot_df["Impact"].max()), abs(plot_df["Impact"].min())) * 0.025
                xp  = val + pad if val >= 0 else val - pad
                ha  = "left" if val >= 0 else "right"
                ax1.text(xp, bar.get_y() + bar.get_height() / 2,
                         f"{val:+.2f}", va="center", ha=ha,
                         fontsize=8.5, color="#b0c4de", fontweight="500",
                         fontfamily="monospace")

            ax1.axvline(0, color="#1e2d45", linewidth=1.2, zorder=0)
            ax1.set_xlabel("SHAP Value  ←  reduces risk  |  increases risk  →",
                           fontsize=9, color="#7b90b2", labelpad=10)
            ax1.tick_params(axis="y", colors="#c8d8f0", labelsize=9.5)
            ax1.tick_params(axis="x", colors="#4a5f7a", labelsize=8)

            for spine in ax1.spines.values():
                spine.set_visible(False)
            ax1.xaxis.grid(True, color="#1a2740", linewidth=0.6, linestyle="--")
            ax1.set_axisbelow(True)

            legend_els = [
                mpatches.Patch(color="#ff4d6d", label="Increases heart disease risk"),
                mpatches.Patch(color="#2f9cf7", label="Reduces heart disease risk"),
            ]
            ax1.legend(handles=legend_els, loc="lower right",
                       fontsize=8.5, framealpha=0,
                       labelcolor="#b0c4de")

            plt.tight_layout(pad=1.2)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            # ── WATERFALL PLOT ──
            st.markdown('<div class="section-label" style="margin-top:1.4rem;">Detailed Prediction Breakdown · Waterfall</div>',
                        unsafe_allow_html=True)

            shap_exp = shap.Explanation(
                values        = sv,
                base_values   = base,
                data          = data_row,
                feature_names = clean_labels
            )

            fig2, ax2 = plt.subplots(figsize=(9, max(7, len(clean_labels) * 0.32)))
            fig2.patch.set_facecolor("#0d1320")

            shap.plots.waterfall(shap_exp, show=False, max_display=12)

            # Re-style the waterfall figure
            for ax in fig2.axes:
                ax.set_facecolor("#0d1320")
                ax.tick_params(colors="#b0c4de", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e2d45")

            plt.tight_layout(pad=1.2)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            # ── CLINICAL INTERPRETATION BOX ──
            risk_drivers  = shap_df[shap_df["Impact"] > 0].sort_values("Impact", ascending=False)
            risk_reducers = shap_df[shap_df["Impact"] < 0].sort_values("Impact", ascending=True)

            drivers_str  = ", ".join(risk_drivers["Feature"].head(3).tolist()) if not risk_drivers.empty else "none"
            reducers_str = ", ".join(risk_reducers["Feature"].head(2).tolist()) if not risk_reducers.empty else "none"

            st.markdown(f"""
            <div class="interp-box">
                <span class="interp-title">🧠 Final AI Interpretation</span>
                The model predicted a <strong>{risk_text.lower()}</strong> of heart disease
                (<strong>{pct:.1f}%</strong>). This prediction is primarily driven by
                <strong>{drivers_str}</strong>, all of which are clinically associated with
                underlying cardiac abnormalities. However,
                <strong>{reducers_str}</strong> partially mitigate the overall risk score.
                The model balances these competing signals to arrive at its final prediction.
            </div>
            """, unsafe_allow_html=True)

            # ── Clinical advisory ──
            if prediction == 1:
                st.markdown("""
                <div style="background:rgba(255,77,109,0.08); border:1px solid rgba(255,77,109,0.25);
                            border-radius:12px; padding:1rem 1.4rem; margin-top:1rem;
                            font-size:0.9rem; color:#ffb3c1;">
                    ⚠️ <strong>Clinical Advisory:</strong> Elevated heart disease likelihood detected.
                    It is recommended to refer this patient to a cardiologist for comprehensive
                    evaluation and confirmatory diagnostic testing.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(45,212,160,0.07); border:1px solid rgba(45,212,160,0.22);
                            border-radius:12px; padding:1rem 1.4rem; margin-top:1rem;
                            font-size:0.9rem; color:#a7f3d0;">
                    ✅ <strong>Clinical Advisory:</strong> Low likelihood of heart disease detected.
                    Encourage continued healthy lifestyle habits, regular monitoring of blood pressure
                    and cholesterol, and routine cardiac check-ups.
                </div>
                """, unsafe_allow_html=True)

            # ── Top Factors Table ──
            st.markdown('<div class="section-label" style="margin-top:1.6rem;">Top Contributing Clinical Factors</div>',
                        unsafe_allow_html=True)

            top_tbl = shap_df.tail(8)[["Feature", "Impact"]].copy()
            top_tbl["Direction"]  = top_tbl["Impact"].apply(
                lambda x: "🔴 Increases Risk" if x > 0 else "🔵 Reduces Risk")
            top_tbl["SHAP Value"] = top_tbl["Impact"].apply(lambda x: f"{x:+.4f}")
            top_tbl = (top_tbl[["Feature", "SHAP Value", "Direction"]]
                       .rename(columns={"Feature": "Clinical Factor"})
                       .sort_values("SHAP Value", ascending=False)
                       .reset_index(drop=True))
            top_tbl.index += 1

            st.dataframe(top_tbl, use_container_width=True, height=310)

            # ── Footer disclaimer ──
            st.markdown("""
            <div style="margin-top:1.5rem; padding:1rem 1.4rem;
                        background:var(--bg-card); border-radius:10px;
                        border:1px solid var(--border);
                        font-family:'Space Mono',monospace; font-size:0.65rem;
                        color:var(--text-muted); line-height:1.7; letter-spacing:0.3px;">
                ⚠ DISCLAIMER · This tool is intended for clinical research and decision
                support purposes only. It does not constitute a medical diagnosis.
                All predictions must be reviewed and validated by a licensed
                healthcare professional before any clinical action is taken.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div style="background:rgba(255,77,109,0.08); border:1px solid rgba(255,77,109,0.3);
                        border-radius:12px; padding:1.4rem; color:#ffb3c1; font-size:0.9rem;">
                ❌ <strong>Prediction Error:</strong> {str(e)}<br><br>
                <span style="color:var(--text-muted); font-size:0.82rem;">
                Ensure <code>xgboost_heart_model.joblib</code> exists in the working directory
                and that all column names match the training data exactly.
                </span>
            </div>
            """, unsafe_allow_html=True)