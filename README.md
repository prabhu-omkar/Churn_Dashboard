# 🎮 Online Gaming Player Churn Prediction Operating System & ML Arena

> 📜 **Academic Research Paper Available**  
> We have published a comprehensive mathematical research paper analyzing why XGBoost dominates tabular churn prediction.  
> 📖 **Read the full research paper here:** [**`Churn_Prediction Repository`**](https://github.com/prabhu-omkar/Churn_Prediction)


---

## 📌 Executive Summary

In the competitive online gaming industry, acquiring new players costs up to **5x to 10x more** than retaining existing ones. Predicting player churn—the permanent disengagement of a player from the game—allows publishers to deploy targeted, preemptive retention campaigns (e.g., personalized daily streak rewards, difficulty adjustments, and special event invites).

This project presents an **end-to-end Machine Learning Operating System** for predicting player churn using **40,034 player records**. The pipeline integrates:
1. **Unsupervised Learning (K-Means Clustering):** Segmenting the player base into macro behavioral personas (Whales, Grinders, Casuals).
2. **Feature Engineering:** Creating domain-specific interaction metrics (`PlayTimePerSession`, `AchievementRate`, `TotalWeeklyMinutes`).
3. **Class Imbalance Resampling (SMOTE):** Synthetic oversampling of the minority churn class strictly within training folds.
4. **Supervised Model Arena:** Comparing Logistic Regression, Random Forest, XGBoost Baseline, and Hyperparameter-Tuned XGBoost.
5. **Explainable AI (SHAP):** Game-theoretic feature attribution providing local and global model interpretability.
6. **Interactive Flask Web Application:** A full-stack web dashboard allowing drag-and-drop CSV upload, automated training, live single/batch inference, and interactive visualization.

---

## 📊 Model Arena Performance Benchmark

All models were evaluated on an untouched **8,007 sample holdout test set** (preserving the real-world 25.8% churn distribution):

| Classifier Model | Accuracy | F1 Score (Churn) | Precision (Churn) | Recall (Churn) | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 84.70% | 0.7501 | 70.81% | 79.76% | 0.9238 | 0.6605 |
| **Random Forest** | 93.71% | 0.8801 | 89.62% | 86.44% | 0.9385 | 0.8377 |
| **XGBoost (Default)** | 94.60% | 0.8957 | 91.02% | 88.16% | 0.9390 | 0.8593 |
| 🏆 **XGBoost (Tuned)** | **95.23%** | **0.9064** | **91.72%** | **89.59%** | **0.9395** | **0.8745** |

### Why XGBoost Won the Arena:
- **Second-Order Taylor Optimization:** Fits Newton steps using both gradients ($g_i$) and Hessians ($h_i$) of binary log-loss.
- **Explicit Leaf Regularization:** Multi-term penalty ($\gamma T + \frac{1}{2}\lambda \sum w_j^2$) prevents overfitting on noisy behavioral metrics.
- **Sequential Error Correction:** Iteratively focuses model capacity on borderline, hard-to-classify players.

---

## 🛠️ Complete Machine Learning Pipeline

```
Raw CSV Data (40,034 rows)
    │
    ├── 1. Data Cleaning & Label Formulation (EngagementLevel == 'Low' -> Churn=1)
    ├── 2. Inferential Statistics (Chi-Square Tests on Categorical Attributes)
    ├── 3. Domain Feature Engineering (5 Derived Behavioral Indicators)
    ├── 4. Unsupervised K-Means Clustering (3 Player Archetypes)
    ├── 5. Preprocessing Pipeline (StandardScaler + OneHotEncoder)
    ├── 6. Imbalance Mitigation (SMOTE synthetic oversampling on training fold)
    ├── 7. Model Arena Training & Grid Search Hyperparameter Tuning
    ├── 8. Evaluation Metrics (Confusion Matrix, ROC-AUC, PR Curves, MCC)
    └── 9. SHAP Explainability (TreeExplainer Beeswarm & Waterfall Attributions)
```

---

## 🧪 Detailed Pipeline Components

### 1. Target Label Derivation & Problem Setup
Since player churn is rarely logged explicitly in passive telemetry datasets, we derived the target variable from `EngagementLevel`:
- **Churned (Class 1):** `EngagementLevel == 'Low'` (25.8% of data)
- **Retained (Class 0):** `EngagementLevel` in `['Medium', 'High']` (74.2% of data)

### 2. Feature Engineering Strategy
We engineered 5 non-linear composite attributes to maximize predictive signal:
- `PlayTimePerSession`: $\frac{\text{PlayTimeHours}}{\text{SessionsPerWeek}}$ (Distinguishes daily habitual users from binge weekend players).
- `AchievementRate`: $\frac{\text{AchievementsUnlocked}}{\text{PlayerLevel} + 1}$ (Detects skill bottlenecks and difficulty progression wall).
- `TotalWeeklyMinutes`: $\text{SessionsPerWeek} \times \text{AvgSessionDurationMinutes}$ (Absolute volume of weekly engagement).
- `IsHighLevel`: Binary flag for players in the top 25th percentile of `PlayerLevel`.
- `AgeGroup`: Binned demographic categories (`Teen`, `Young Adult`, `Adult`, `Senior`).

### 3. Statistical Testing ($\chi^2$ Independence Tests)
Pearson Chi-Square tests confirmed that demographic variables (`Gender`, `Location`) and subjective preferences (`GameGenre`, `GameDifficulty`) have **$p$-values > 0.30** (statistically insignificant relationship with churn). Churn is driven entirely by **continuous behavioral activity**.

### 4. Unsupervised K-Means Player Segmentation
Using normalized behavioral features, K-Means clustering ($K=3$) identified 3 core player archetypes:
1. 🦈 **Whales (High Spenders):** Low-to-medium playtime, but massive spikes on `InGamePurchases`.
2. ⚔️ **Hardcore Grinders:** Maximum `PlayTimeHours`, high `SessionsPerWeek`, high `AchievementsUnlocked`. Low churn rate.
3. 🚶 **Casual Players:** Minimal sessions, short session durations, low unlock count. **High Churn Risk (Primary target for retention campaigns).**

### 5. SMOTE Imbalance Handling
To prevent model bias toward the 74.2% majority class, we applied **Synthetic Minority Over-sampling Technique (SMOTE)**:
- **Training Set (Pre-SMOTE):** 32,027 samples (imbalanced 74/26 ratio).
- **Training Set (Post-SMOTE):** 47,536 samples (perfect 50/50 balance).
- *Strict Leakage Prevention:* SMOTE was applied strictly to the training fold; the test set (8,007 samples) remained untouched.

---

## 🔍 Model Explainability with SHAP (TreeExplainer)

Machine learning models should not operate as "black boxes." We integrated **SHAP (SHapley Additive exPlanations)** based on cooperative game theory to quantify feature contributions:

$$\phi_j = \sum_{S \subseteq \mathcal{N} \setminus \{j\}} \frac{|S|!\,(|\mathcal{N}| - |S| - 1)!}{|\mathcal{N}|!} \left[f_{S \cup \{j\}}(x) - f_S(x)\right]$$

### Key SHAP Findings:
1. **`SessionsPerWeek` (Primary Driver):** Low login frequency is the single strongest indicator pushing predictions toward Churn ($y=1$). A sharp churn cliff occurs when sessions drop below **3 per week**.
2. **`AvgSessionDurationMinutes`:** Short session length is the second strongest predictor, signaling rapid disengagement.
3. **`TotalWeeklyMinutes`:** Validates feature engineering by adding composite predictive power.
4. **Demographics:** SHAP values for `Gender`, `Location`, and `GameGenre` are near zero, proving the model correctly ignored non-predictive attributes.

---

## 🌐 Interactive Web Dashboard (Flask ML OS)

The codebase includes a web application (`app.py`) featuring:
- 📤 **CSV Upload & Preprocessing:** Instant handling of custom player datasets.
- 🏋️ **Auto-Training Pipeline:** Triggers data preprocessing, SMOTE, model training, and SHAP computation.
- 🎯 **Model Arena Tab:** Comparative view of Accuracy, Precision, Recall, F1, ROC curves, and Confusion Matrices.
- 🔮 **Single Player & Batch Inference:** Input individual player metrics to get instant churn probability scores and personalized retention recommendations.
- 📊 **Persona Radar Visualizer:** Interactive radar plots showing cluster centroids.

---

## 💻 Installation & Usage

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/prabhu-omkar/Churn_Prediction.git
cd Churn_Prediction

# Install requirements
pip install -r requirements.txt
# (or pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn imbalanced-learn flask)
```

### 2. Run the Machine Learning Pipeline Script
```bash
python churn_analysis.py
```
*Outputs model performance metrics, JSON logs, and saves publication plots in `plots/`.*

### 3. Launch the Web Application
```bash
python app.py
```
*Open your browser and navigate to `http://127.0.0.1:5000` to interact with the dashboard.*

---

## 📂 Repository Structure

```
Churn_Dashboard/
├── README.md                           # Main project documentation
├── app.py                              # Flask Web Application & Model Arena OS
├── requirements.txt                    # Python package dependencies
├── Procfile                            # Production WSGI server start command
├── render.yaml                         # Render 1-click deployment blueprint
├── online_gaming_behavior_dataset.csv  # Sample dataset (40,034 player records)
├── static/                             # Web CSS styles & dynamic uploads
└── templates/                          # HTML dashboard interface (index.html)
```

---

## 📖 Citation & Research Paper

If you use this repository or research paper in your work, please cite:

```bibtex
@article{churn_xgboost_2026,
  title={Why XGBoost Dominates Tabular Churn Prediction: A Mathematical and Empirical Analysis},
  author={Churn Prediction OS Team},
  journal={Online Gaming Analytics Repository},
  year={2026},
  url={https://github.com/prabhu-omkar/Churn_Prediction}
}
```

*Read the full mathematical derivations and formal proofs in the [Churn_Prediction Repository](https://github.com/prabhu-omkar/Churn_Prediction).*

