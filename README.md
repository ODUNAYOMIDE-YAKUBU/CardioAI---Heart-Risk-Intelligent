<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=CardioAI%20%F0%9F%AB%80&fontSize=60&fontColor=ffffff&fontAlignY=35&desc=Explainable%20AI%20for%20Heart%20Disease%20Prediction&descAlignY=58&descSize=20&descColor=a0c4ff" width="100%"/>

<!-- BADGES ROW 1 -->
<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-E63946?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

<!-- BADGES ROW 2 -->
<p>
  <img src="https://img.shields.io/badge/AUC--ROC-0.9207-2dd4a0?style=flat-square"/>
  <img src="https://img.shields.io/badge/Accuracy-88.0%25-2f9cf7?style=flat-square"/>
  <img src="https://img.shields.io/badge/Recall%20(Disease%2B)-88%25-f7b731?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/github/stars/yourusername/cardioai?style=flat-square&color=yellow"/>
</p>

<br/>

> **🏥 The first XAI-integrated cardiovascular ML system contextualised for Nigerian and sub-Saharan African healthcare.**  
> Predicts heart disease with clinical-grade accuracy — and *explains every prediction* using SHAP.

<br/>

[**🚀 Live Demo**](#-live-demo) · [**📖 Documentation**](#-how-it-works) · [**⚡ Quick Start**](#-quick-start) · [**📊 Results**](#-model-performance) · [**🤝 Contributing**](#-contributing)

<br/>

</div>

---

## 🌍 Why This Project Exists

Cardiovascular disease kills **17.9 million people annually** — and over **75%** of those deaths happen in countries like Nigeria, where cardiologist access is critically limited and diagnostic infrastructure is scarce.

Machine learning can predict heart disease from routine clinical data. But standard ML models are **black boxes** — they give a prediction with no explanation. In healthcare, that's a dealbreaker. Doctors won't (and shouldn't) act on a decision they can't understand.

**CardioAI solves this** by combining:
- ✅ A high-accuracy **XGBoost** classifier (AUC = 0.92)
- ✅ **SHAP** explainability that shows *exactly* which clinical features drove each prediction
- ✅ A **Streamlit** web app that presents results in plain clinical language — no ML background required

---

## 🖥️ Live Demo

<div align="center">

| Input Panel | SHAP Explanation | Clinical Interpretation |
|:-----------:|:----------------:|:----------------------:|
| ![Input](docs/images/app_input.png) | ![SHAP Bar](docs/images/shap_bar.png) | ![Interpretation](docs/images/interpretation.png) |
| 13 clinical parameters | Feature contribution chart | Natural language risk summary |

</div>

> 🔗 **Deploy your own instance** → see [Quick Start](#-quick-start) below

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 Machine Learning
- **5 classifiers** trained & compared under identical conditions
- **XGBoost** pipeline with `ColumnTransformer` preprocessing
- `StandardScaler` + `OneHotEncoder` — no data leakage
- Stratified 80/20 train-test split
- Model serialised as `.joblib` for instant inference

</td>
<td width="50%">

### 🧠 Explainability (XAI)
- **SHAP TreeExplainer** — exact Shapley values for every prediction
- **Global analysis** — population-level feature importance
- **Local analysis** — patient-specific waterfall & bar plots
- **Natural language** interpretation auto-generated per patient
- Clean, human-readable feature labels (no `cat__` prefixes)

</td>
</tr>
<tr>
<td width="50%">

### 🏥 Clinical Interface
- **Streamlit** web app — zero front-end code needed
- Colour-coded risk meter (🟢 Low / 🟡 Moderate / 🔴 High)
- SHAP waterfall + bar chart rendered inline
- Ranked contributing factors table
- Clinical advisory message per patient

</td>
<td width="50%">

### 🏗️ System Architecture
- **3-layer design**: Presentation → Application → AI/ML
- Modular & fully reproducible pipeline
- 100% open-source stack
- Runs on standard hardware — no GPU needed
- Deployable in resource-constrained settings

</td>
</tr>
</table>

---

## 📊 Model Performance

### Comparative Evaluation — All 5 Classifiers

| Model | Accuracy | AUC-ROC |
|:------|:--------:|:-------:|
| SVM | 88.59% | 0.9144 |
| KNN | 88.04% | 0.9191 |
| Random Forest | 88.04% | 0.9191 |
| Logistic Regression | 86.96% | 0.9114 |
| **Gradient Boosting** | 85.87% | **0.9223** ⬅ best AUC |

> Gradient Boosting achieved the highest AUC-ROC, motivating the selection of **XGBoost** (its advanced, regularised counterpart) as the final deployed model.

### Final XGBoost Model — Detailed Report

| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|:-------:|
| No Heart Disease (0) | 0.84 | 0.88 | 0.86 | 75 |
| Heart Disease (1) | **0.91** | **0.88** | **0.90** | 109 |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | **184** |
| **AUC-ROC** | — | — | **0.9207** | — |

> 💡 **88% recall on the disease-positive class** — 88 out of every 100 true heart disease cases correctly identified.

---

## 🧠 How It Works

```
Patient Clinical Data (13 features)
          │
          ▼
┌─────────────────────────────┐
│   Presentation Layer        │  ← Streamlit Web App
│   Patient Input Form        │
│   Risk Display + SHAP Viz   │
└────────────┬────────────────┘
             │ User Input & Output
             ▼
┌─────────────────────────────┐
│   Application Layer         │  ← Backend Logic
│   Input Validation          │
│   Feature Alignment         │
│   SHAP Integration          │
└────────────┬────────────────┘
             │ Processed Data Flow
             ▼
┌──────────────────────────────────────────────┐
│   AI/ML Layer  (Core Intelligence Engine)    │
│                                              │
│  Data Processing → Model Training            │
│  ColumnTransformer → XGBoost Pipeline        │
│  ┌──────────────────────────────────────┐    │
│  │  SHAP (TreeExplainer)  ← KEY MODULE  │    │
│  │  Feature Importance                  │    │
│  │  Local Prediction · SHAP Values      │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
             │
             ▼
    Prediction + Explanation
```

---

## 🔬 SHAP Explainability in Action

### Global Feature Importance
The most influential clinical features across all patients:

| Rank | Feature | Direction | Avg SHAP Impact |
|:----:|:--------|:---------:|:---------------:|
| 1 | Thalassemia: Normal | ↑ Increases Risk | +0.95 |
| 2 | Chest Pain: Asymptomatic | ↑ Increases Risk | +0.64 |
| 3 | ST Slope: Flat | ↓ Reduces Risk | −0.48 |
| 4 | Thalassemia: Fixed Defect | ↑ Increases Risk | +0.33 |
| 5 | Thalassemia: Reversible Defect | ↓ Reduces Risk | −0.28 |
| 6 | Resting Blood Pressure | ↓ Reduces Risk | −0.27 |
| 7 | Exercise-Induced Angina: No | ↓ Reduces Risk | −0.25 |

### Local Explanation Example
> For a patient with **27.3% predicted risk (LOW RISK)**:
>
> *"The model predicted a low risk of heart disease (27.3%). This prediction is primarily driven by Chest Pain: Asymptomatic (−0.99) and Serum Cholesterol (−0.83), which strongly reduce the risk score. However, Thalassemia: Normal (+0.78) and Thalassemia: Fixed Defect (+0.34) partially counteract this reduction. The model balances these competing signals to arrive at its final prediction."*

---

## 🗂️ Project Structure

```
cardioai/
│
├── 📓 notebooks/
│   └── heart_disease_model.ipynb     # Full training pipeline & evaluation
│
├── 🌐 app/
│   └── app.py                        # Streamlit web application
│
├── 🤖 models/
│   └── xgboost_heart_model.joblib    # Trained & serialised XGBoost pipeline
│
├── 📊 data/
│   └── heart_disease_cleaned.csv     # Preprocessed UCI Cleveland dataset
│
├── 📁 docs/
│   └── images/                       # Screenshots & figures
│
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ODUNAYOMIDE-YAKUBU/cardioai.git
cd cardioai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model *(or skip — pretrained model included)*
```bash
jupyter notebook notebooks/heart_disease_model.ipynb
```

### 4. Launch the Streamlit app
```bash
streamlit run app/app.py
```

> The app will open at `http://localhost:8501` 🚀

---

## 📦 Requirements

```txt
streamlit>=1.28.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
joblib>=1.3.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

This project uses the **UCI Heart Disease Dataset (Cleveland subset)** — the most widely validated benchmark for cardiovascular ML research.

| Property | Value |
|:---------|:------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Records | 303 patients |
| Features | 13 clinical predictors |
| Target | Binary (0 = No Disease, 1 = Disease) |
| Class Ratio | 54.5% negative / 45.5% positive |

**Features include:** Age, Sex, Chest Pain Type, Resting BP, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, ST Depression, ST Slope, No. of Major Vessels, Thalassemia

---

## 🏗️ System Architecture

<div align="center">
<img src="docs/images/system_architecture.png" width="80%" alt="System Architecture"/>
</div>

The system is organised into three layers:
- **Presentation Layer** — Streamlit web app with patient input form, risk level display, and SHAP visualisations (waterfall + bar chart)
- **Application Layer** — Input validation, feature alignment, prediction handling, and SHAP integration
- **AI/ML Layer** — ColumnTransformer preprocessing → XGBoost classifier → SHAP TreeExplainer

---

## 📚 Research Context

This project was developed as academic research addressing a critical gap in the literature:

> **No prior study from Nigeria or West Africa had combined ML-based heart disease prediction with SHAP explainability.**

### Contributions to Knowledge
1. 🇳🇬 First XAI-integrated cardiovascular ML study from Nigeria / West Africa
2. 📐 Empirical proof that accuracy and transparency are simultaneously achievable in clinical AI
3. 🔁 Fully replicable open-source XGBoost-SHAP pipeline for LMIC healthcare settings
4. ✅ Clinical validation of SHAP feature rankings against established cardiology literature
5. 🖥️ First functional XAI cardiac risk prototype designed for the Nigerian healthcare context

### Related Literature
- Muhammad et al. (2021) — ML for coronary artery disease in Nigeria *(no XAI)*
- Shrestha (2024) — XGBoost + SHAP on Cleveland dataset (AUC = 0.94)
- XAI-HD Framework, Springer Nature (2025) — hybrid ML/XAI for heart disease
- Lundberg & Lee (2017) — Original SHAP paper

---

## 🛣️ Roadmap

- [x] Multi-classifier comparative training & evaluation
- [x] XGBoost best-model pipeline with SHAP
- [x] Streamlit prototype with human-readable SHAP labels
- [x] Natural language clinical interpretation layer
- [ ] Primary Nigerian clinical dataset integration
- [ ] LIME complementary explainability layer
- [ ] Counterfactual explanations ("What would change this prediction?")
- [ ] EHR system API integration
- [ ] Docker containerisation for easy deployment
- [ ] Multi-disease extension (diabetes, stroke, CKD)

---

## 🤝 Contributing

Contributions are welcome and encouraged — especially from researchers working in African healthcare AI.

```bash
# Fork the repo
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting. All contributors will be acknowledged.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.  
You are free to use, modify, and distribute this project with attribution.

---

## 📬 Contact & Citation

**Author:** Odunayomide YAKUBU    
**Email:** usyak12@gmail.com  

If you use this project in your research, please cite:

```bibtex
@project{cardioai2025,
  title     = {Design and Implementation of an Explainable AI Model
               for Heart Disease Prediction in Healthcare Systems},
  author    = {[Odunayomide YAKUBU]},
  year      = {2026},
  school    = {[Your University]},
  note      = {Available at: https://github.com/ODUNAYOMIDE-YAKUBU/cardioai}
}
```

---

<div align="center">

**⭐ If this project helped you, please star the repo — it helps others find it!**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

<sub>Built with ❤️ for better healthcare AI in Nigeria and across sub-Saharan Africa</sub>

</div># CardioAI---Heart-Risk-Intelligent
🫀 An Explainable AI system for heart disease prediction using XGBoost + SHAP, with a Streamlit web interface — built for clinical decision support in resource-limited healthcare settings.
