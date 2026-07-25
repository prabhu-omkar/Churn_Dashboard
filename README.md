# 🎮 Online Gaming Player Churn Prediction Operating System & Web Dashboard

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-00b4d8?style=for-the-badge&logo=render)](https://churn-dashboard-nkxv.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost_2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-7209b7?style=for-the-badge)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### 🌐 **[Click Here to Launch Live Interactive Dashboard](https://churn-dashboard-nkxv.onrender.com/)**

</div>

---

## 📌 Executive Summary

In the highly competitive online gaming industry, acquiring a new player costs **5x to 10x more** than retaining an existing user. Predicting player churn—the permanent disengagement of a player from the game—enables publishers and analytics teams to deploy targeted, preemptive retention campaigns (e.g., daily streak rewards, difficulty adjustments, and exclusive event passes).

This repository presents an **end-to-end Machine Learning Operating System & Interactive Web Dashboard** deployed on cloud infrastructure. Built on **40,034 player telemetry records**, the application provides real-time automated model training, player persona clustering, SHAP feature explainability, and live single-player risk inference.

> 📖 **Academic Research Paper Available**:  
> For full mathematical derivations, statistical testing, and proofs on why XGBoost dominates tabular churn prediction, read our research paper repository:  
> 👉 [**`prabhu-omkar/Churn_Prediction`**](https://github.com/prabhu-omkar/Churn_Prediction)

---

## 🚀 Key Features

* ⚡ **Ultra-Fast Machine Learning Engine**: Trains high-accuracy XGBoost models in **~1–2 seconds** using histogram-based tree building (`tree_method='hist'`).
* 📤 **Universal CSV Data Ingestion**: Drag-and-drop custom dataset upload with automatic column type inference, handling for categorical/numeric targets, and automatic dataset fallback.
* 📊 **Unsupervised Persona Clustering**: Segment player behavior into 3 core personas (*Whales*, *Hardcore Grinders*, *Casual Players*) via K-Means ($K=3$).
* ⚖️ **Adaptive Class Balancing**: Synthetic Minority Over-sampling Technique (`SMOTE`) prevents majority-class bias without test data leakage.
* 🔍 **Explainable AI (SHAP)**: Cooperative game-theoretic feature attributions via `shap.TreeExplainer` providing global beeswarm & bar plots.
* 🔮 **Live Single-Player Risk Simulator**: Interactive dashboard tab allowing game designers to input player attributes and obtain instant churn probability scores.
* 🌐 **Production-Ready Deployment**: Configured for WSGI Gunicorn deployment with `Procfile` and `render.yaml` infrastructure-as-code.

---

## 📊 Model Arena Performance Benchmark

All models were evaluated on an untouched **8,007 sample holdout test set** (preserving the real-world 25.8% churn distribution):

| Classifier Model | Accuracy | F1 Score (Churn) | Precision (Churn) | Recall (Churn) | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 84.70% | 0.7501 | 70.81% | 79.76% | 0.9238 | 0.6605 |
| **Random Forest** | 93.71% | 0.8801 | 89.62% | 86.44% | 0.9385 | 0.8377 |
| **XGBoost (Default)** | 94.60% | 0.8957 | 91.02% | 88.16% | 0.9390 | 0.8593 |
| 🏆 **XGBoost (Optimized OS)** | **95.23%** | **0.9064** | **91.72%** | **89.59%** | **0.9395** | **0.8745** |

### Why XGBoost Won the Arena:
1. **Second-Order Taylor Optimization**: Fits Newton steps using both gradients ($g_i$) and Hessians ($h_i$) of binary log-loss.
2. **Explicit Leaf Regularization**: Multi-term penalty ($\gamma T + \frac{1}{2}\lambda \sum w_j^2$) prevents overfitting on noisy telemetry.
3. **Sequential Error Correction**: Boosting iteratively focuses capacity on borderline, high-risk churn players.

---

## 🛠️ System Architecture & ML Pipeline

```
┌────────────────────────┐
│  Raw CSV Telemetry     │ (40,034 Player Records / Custom Upload)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 1. Schema Processing   │ Auto-Detect Target Column & Feature Types
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 2. Feature Engineering │ Compute Derived Signals (PlayTimePerSession, AchievementRate)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 3. Player Segmentation │ K-Means Clustering (K=3) for Persona Radar Generation
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 4. Resampling (SMOTE)  │ Synthetic Over-sampling on Training Fold
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 5. XGBoost Model Fit   │ Histogram Tree Building (`tree_method='hist'`)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 6. SHAP Interpretations│ TreeExplainer Feature Attribution & Global Beeswarm Plots
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ 7. Flask Dashboard UI  │ Interactive Metrics, Radar Charts & Single-Player Risk Inference
└────────────────────────┘
```

---

## 🔍 Model Explainability (SHAP TreeExplainer)

Machine learning models operating in production cannot be black boxes. We integrated **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:

$$\phi_j = \sum_{S \subseteq \mathcal{N} \setminus \{j\}} \frac{|S|!\,(|\mathcal{N}| - |S| - 1)!}{|\mathcal{N}|!} \left[f_{S \cup \{j\}}(x) - f_S(x)\right]$$

### Key Insights Identified by SHAP:
* 📉 **`SessionsPerWeek` (Primary Driver)**: Low login frequency is the single strongest indicator pushing predictions toward Churn ($y=1$). A sharp churn cliff occurs when logins drop below **3 sessions per week**.
* ⏱️ **`AvgSessionDurationMinutes`**: Short session length is the second strongest predictor, signaling rapid player fatigue.
* 📈 **`TotalWeeklyMinutes`**: Validates interaction feature engineering by adding composite predictive signal.
* 🛡️ **Demographic Independence**: SHAP attributions for `Gender`, `Location`, and `GameGenre` remain near zero, confirming the model ignores non-predictive attributes.

---

## 🌐 Live Web Application Routes

The Flask backend ([`app.py`](file:///c:/projects/vidoe-game_churn-prediction/web%20dashboard/app.py)) exposes clean JSON API endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Renders the primary single-page retro OS web dashboard interface. |
| `/upload` | `POST` | Handles drag-and-drop CSV dataset upload, extracts columns & previews. |
| `/analyze` | `POST` | Triggers the complete ML pipeline, trains XGBoost, generates plots & metrics. |
| `/simulate` | `POST` | Accepts single-player feature JSON & returns real-time churn probability. |

---

## 💻 Local Installation & Setup

### Prerequisites
* Python `3.10` or higher
* Git

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/prabhu-omkar/Churn_Dashboard.git
   cd Churn_Dashboard
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Local Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to: **`http://localhost:5000`**

---

## ☁️ Deployment Instructions (Render)

This repository includes pre-configured infrastructure files for seamless deployment to **[Render](https://render.com)**:

1. **`Procfile`**: Defines the production WSGI start command (`web: gunicorn app:app`).
2. **`render.yaml`**: Infrastructure-as-code blueprint for Render Web Services.

### 1-Click Deployment to Render:
1. Fork or push this repository to GitHub.
2. Sign in to **[Render.com](https://render.com)**.
3. Click **New +** → **Web Service**.
4. Select **`prabhu-omkar/Churn_Dashboard`**.
5. Click **Create Web Service**. Your app will be live automatically!

---

## 📂 Repository Layout

```
Churn_Dashboard/
├── README.md                           # Comprehensive documentation (You are here)
├── app.py                              # Flask Web Application & ML Pipeline Engine
├── requirements.txt                    # Production Python package dependencies
├── Procfile                            # WSGI Gunicorn launch configuration
├── render.yaml                         # Render Cloud Deployment Blueprint
├── online_gaming_behavior_dataset.csv  # Primary telemetry dataset (40,034 records)
├── static/                             # Retro OS CSS stylesheet & dynamic upload cache
│   ├── style.css
│   └── uploads/
└── templates/                          # Single-page web dashboard (index.html)
    └── index.html
```

---

## 📖 Citation & Research Paper

If you use this project, dashboard, or dataset in your research or commercial applications, please cite:

```bibtex
@article{churn_xgboost_2026,
  title={Why XGBoost Dominates Tabular Churn Prediction: A Mathematical and Empirical Analysis},
  author={Omkar, Prabhu and Churn Prediction OS Team},
  journal={Online Gaming Analytics Repository},
  year={2026},
  url={https://github.com/prabhu-omkar/Churn_Prediction}
}
```

---

<div align="center">

Made with ❤️ for Game Analytics & Machine Learning Engineering  
**Deployed App**: [https://churn-dashboard-nkxv.onrender.com/](https://churn-dashboard-nkxv.onrender.com/)

</div>
