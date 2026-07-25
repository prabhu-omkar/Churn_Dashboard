"""
CHURN PREDICTION OS - Interactive ML Pipeline
Upload CSV -> Auto-train -> Visualize -> Explain -> Predict
"""
import os, uuid, json, traceback, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from math import pi
from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             f1_score, accuracy_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Theme
BG = "#f4eefa"; CARD = "#ffffff"; NEON_CYAN = "#00b4d8"; NEON_PINK = "#f72585"
NEON_PURPLE = "#7209b7"; GOLD = "#ff9f1c"; GRID_COLOR = "#d5c6e0"; TEXT = "#2b1c3d"
PALETTE = [NEON_CYAN, NEON_PINK, NEON_PURPLE, "#06d6a0", GOLD, "#f4a261"]

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'axes.grid': True, 'grid.color': GRID_COLOR, 'grid.alpha': 0.3,
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})



models_store = {}

def apply_feature_engineering(X, state=None):
    if state is None: state = {}
    cols_lower = {c: c.lower() for c in X.columns}
    col_map = {v: k for k, v in cols_lower.items()}
    if 'playtimehours' in col_map and 'sessionsperweek' in col_map:
        X['PlayTimePerSession'] = np.where(X[col_map['sessionsperweek']] > 0, X[col_map['playtimehours']] / X[col_map['sessionsperweek']], 0.0)
    if 'achievementsunlocked' in col_map and 'playerlevel' in col_map:
        X['AchievementRate'] = np.where(X[col_map['playerlevel']] > 0, X[col_map['achievementsunlocked']] / X[col_map['playerlevel']], 0.0)
    if 'sessionsperweek' in col_map and 'avgsessiondurationminutes' in col_map:
        X['TotalWeeklyMinutes'] = X[col_map['sessionsperweek']] * X[col_map['avgsessiondurationminutes']]
    if 'playerlevel' in col_map:
        lv_col = col_map['playerlevel']
        if 'level_q75' not in state: state['level_q75'] = X[lv_col].quantile(0.75)
        X['IsHighLevel'] = (X[lv_col] >= state['level_q75']).astype(int)
    if 'age' in col_map:
        X['AgeGroup'] = pd.cut(X[col_map['age']], bins=[0, 18, 25, 35, 50], labels=['Teen', 'Young Adult', 'Adult', 'Senior'])
        X['AgeGroup'] = X['AgeGroup'].astype(str).replace('nan', 'Senior')
    num_only = X.select_dtypes(include=[np.number]).columns
    X[num_only] = X[num_only].replace([float('inf'), -float('inf')], 0).fillna(0)
    return X, state

def save_plot(fig, plot_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, name), dpi=100, facecolor=BG)
    plt.close(fig)
    return name



def run_pipeline(df, target_col, positive_val, sid):
    """Full ML pipeline: cluster, train, evaluate, explain."""
    plot_dir = "web_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plots = []
    
    # --- 1. TARGET ---
    # Build binary target robustly
    is_string_col = (df[target_col].dtype == 'object' or 
                     pd.api.types.is_string_dtype(df[target_col]) or
                     not pd.api.types.is_numeric_dtype(df[target_col]))
    
    if positive_val and str(positive_val).strip():
        y = (df[target_col].astype(str).str.strip() == str(positive_val).strip()).astype(int)
    elif is_string_col:
        # String column but no positive_val provided - pick the least frequent class as positive
        vc = df[target_col].value_counts()
        minority = vc.index[-1]
        y = (df[target_col] == minority).astype(int)
    else:
        # Numeric column
        uniq = df[target_col].nunique()
        if uniq == 2:
            y = df[target_col].astype(int)
        else:
            # Use median split for continuous targets
            median_val = df[target_col].median()
            y = (df[target_col] >= median_val).astype(int)

    # Validate binary target
    n_classes = y.nunique()
    if n_classes != 2:
        raise ValueError(f"Target must be binary (2 classes). Got {n_classes} classes. Please select a valid target column and positive class value.")
    min_class_count = y.value_counts().min()
    if min_class_count < 10:
        raise ValueError(f"Smallest class has only {min_class_count} samples. Need at least 10 for reliable training.")

    drop_cols = [target_col]
    for c in df.columns:
        if c.lower().endswith('id') or c.lower() == 'id':
            drop_cols.append(c)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Extract raw feature ranges for simulator BEFORE engineering
    simulator_config = {'numeric': {}, 'categorical': {}}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            simulator_config['numeric'][c] = {
                'min': float(X[c].min()), 'max': float(X[c].max()), 'mean': float(X[c].mean())
            }
        else:
            simulator_config['categorical'][c] = X[c].astype(str).unique().tolist()

    # --- FEATURE ENGINEERING ---
    X, state = apply_feature_engineering(X)

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    info = {
        'rows': len(df), 'features': len(X.columns),
        'num_cols': num_cols, 'cat_cols': cat_cols,
        'churn_rate': round(float(y.mean()), 4),
        'engineered_features': [c for c in ['PlayTimePerSession', 'AchievementRate', 'TotalWeeklyMinutes', 'IsHighLevel', 'AgeGroup'] if c in X.columns],
        'simulator': simulator_config
    }

    # --- 2. CLUSTERING ---
    if len(num_cols) >= 2:
        cluster_feats = num_cols[:6]
        sc = MinMaxScaler()
        Xc = sc.fit_transform(X[cluster_feats].fillna(0))
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(Xc)


    # --- 5. PREPROCESSING ---
    transformers = []
    if num_cols: transformers.append(('num', StandardScaler(), num_cols))
    if cat_cols: transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols))
    preprocessor = ColumnTransformer(transformers)
    Xp = np.nan_to_num(preprocessor.fit_transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    feat_names = list(num_cols)
    if cat_cols:
        feat_names += list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))

    Xtr, Xte, ytr, yte = train_test_split(Xp, y, test_size=0.2, random_state=42, stratify=y)
    minority_count = pd.Series(ytr).value_counts().min()
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)


    # --- 6. TRAIN XGBOOST (Fast & High Accuracy) ---
    spw = max((len(ytr)-sum(ytr))/max(sum(ytr),1), 1)
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        tree_method='hist', max_depth=6, learning_rate=0.08, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        random_state=42, n_jobs=-1
    )
    model.fit(Xtr_s, ytr_s)

    yp = model.predict(Xte)
    ypr = model.predict_proba(Xte)[:, 1]

    metrics = {
        'accuracy': round(float(accuracy_score(yte, yp)), 4),
        'f1': round(float(f1_score(yte, yp)), 4),
        'auc': round(float(roc_auc_score(yte, ypr)), 4),
        'mcc': round(float(matthews_corrcoef(yte, yp)), 4),
        'best_params': {'max_depth': 6, 'learning_rate': 0.08, 'n_estimators': 150},
        'report': classification_report(yte, yp, target_names=['Retained', 'Churned'], output_dict=True)
    }


    # --- PLOTS ---
    # 01 Persona Radar & 04 Persona Churn
    if len(num_cols) >= 2:
        # Radar
        categories = cluster_feats
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(BG)
        for i, (color, ax) in enumerate(zip([NEON_CYAN, NEON_PINK, GOLD], axes)):
            ax.set_facecolor(BG)
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, color=TEXT, size=10)
            ax.tick_params(axis='x', pad=15)
            ax.set_rlabel_position(0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels([], color=GRID_COLOR, size=8)
            ax.set_ylim(0, 1)
            ax.grid(color=GRID_COLOR, linestyle='--', linewidth=1, alpha=0.7)
            values = km.cluster_centers_[i].tolist()
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid')
            ax.fill(angles, values, color=color, alpha=0.5)
            ax.set_title(f"Persona {i}", size=14, color=color, fontweight='bold', y=1.15)
        fig.suptitle("Player Personas", size=18, color=TEXT, y=1.05, fontweight='bold')
        plt.tight_layout()
        plots.append(save_plot(fig, plot_dir, "01_persona_radar.png"))

        # Churn
        cdf = pd.DataFrame({'Cluster': labels, 'Churn': y.values})
        agg = cdf.groupby('Cluster')['Churn'].agg(['mean', 'count']).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar([f"Persona {i}" for i in agg['Cluster']], agg['mean'], color=[NEON_CYAN, NEON_PINK, GOLD], edgecolor=BG, width=0.5)
        for bar, r, c in zip(bars, agg['mean'], agg['count']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{r:.1%}\n(n={c:,})',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=TEXT)
        ax.set_title("Churn Rate by Segment", fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(agg['mean'])*1.4)
        plots.append(save_plot(fig, plot_dir, "04_persona_churn_rate.png"))

    # 02 Correlation Heatmap
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        cdata = X[num_cols].copy(); cdata['Target'] = y.values
        corr = cdata.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = LinearSegmentedColormap.from_list("n", [NEON_PINK, BG, NEON_CYAN])
        sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f', center=0, ax=ax,
                    linewidths=0.5, linecolor=GRID_COLOR, annot_kws={'size': 9})
        ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
        plots.append(save_plot(fig, plot_dir, "02_correlation_heatmap.png"))
        
        # 03 Feature Distributions (Generic for any dataset)
        dist_features = ['PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
                         'TotalWeeklyMinutes', 'PlayerLevel', 'AchievementsUnlocked']
        dist_features = [f for f in dist_features if f in X.columns]
        if not dist_features and num_cols:
            dist_features = num_cols[:6]

        if dist_features:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for i, col in enumerate(dist_features):
                ax = axes.flatten()[i]
                sns.histplot(x=X[col], hue=y, bins=30, ax=ax, 
                             palette=[NEON_CYAN, NEON_PINK], edgecolor=BG, alpha=0.7)
                ax.set_title(col, fontsize=12, fontweight='bold')
                ax.set_xlabel('')
                
            for j in range(len(dist_features), len(axes.flatten())):
                axes.flatten()[j].set_visible(False)
                
            fig.suptitle("Key Feature Distributions by Target Class", fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            plots.append(save_plot(fig, plot_dir, "03_feature_distributions.png"))




    # 07 ROC
    fig, ax = plt.subplots(figsize=(10,8))
    fpr, tpr, _ = roc_curve(yte, ypr)
    ax.plot(fpr, tpr, color=NEON_CYAN, lw=2, label=f'XGBoost (AUC={metrics["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend()
    ax.set_title("ROC Curve", fontsize=16, fontweight='bold')
    plots.append(save_plot(fig, plot_dir, "07_roc_curves.png"))

    # 08 PR
    fig, ax = plt.subplots(figsize=(10,8))
    pr, rc, _ = precision_recall_curve(yte, ypr)
    ap = average_precision_score(yte, ypr)
    ax.plot(rc, pr, color=NEON_PINK, lw=2, label=f'XGBoost (AP={ap:.4f})')
    ax.legend()
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight='bold')
    plots.append(save_plot(fig, plot_dir, "08_precision_recall.png"))

    # 09 Confusion
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(yte, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained','Churned'], yticklabels=['Retained','Churned'],
                linewidths=1, linecolor=GRID_COLOR, cbar=False, annot_kws={'size': 18, 'fontweight': 'bold'})
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    plots.append(save_plot(fig, plot_dir, "09_confusion_matrices.png"))

    # 10 Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    imp = model.feature_importances_
    si = np.argsort(imp)[-19:]
    ax.barh(range(len(si)), imp[si], color=NEON_CYAN, edgecolor=BG, alpha=0.85)
    ax.set_yticks(range(len(si)))
    ax.set_yticklabels([feat_names[i] for i in si], fontsize=11)
    ax.set_ylim(-0.5, len(si) - 0.5)
    ax.set_title('Top Feature Importances', fontsize=16, fontweight='bold')
    plots.append(save_plot(fig, plot_dir, "10_feature_importance.png"))

    # 11 & 12 SHAP (Fast Subsampled Evaluation)
    Xte_shap = Xte[:500] if len(Xte) > 500 else Xte
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(Xte_shap)
    fig = plt.figure(figsize=(10, 14))
    shap.summary_plot(shap_vals, Xte_shap, feature_names=feat_names, show=False, max_display=19, plot_size=(10, 14))
    plt.gcf().patch.set_facecolor(BG); plt.gca().set_facecolor(BG); plt.gca().tick_params(colors=TEXT)
    plots.append(save_plot(fig, plot_dir, "11_shap_summary.png"))
    
    fig, ax = plt.subplots(figsize=(10, 14))
    shap.summary_plot(shap_vals, Xte_shap, feature_names=feat_names, plot_type='bar', show=False, color=NEON_PURPLE, max_display=19, plot_size=(10, 14))
    plt.gcf().patch.set_facecolor(BG); plt.gca().set_facecolor(BG); plt.gca().tick_params(colors=TEXT)
    plots.append(save_plot(plt.gcf(), plot_dir, "12_shap_bar.png"))



    models_store[sid] = {'model': model, 'preprocessor': preprocessor, 'state': state, 'raw_columns': list(simulator_config['numeric'].keys()) + list(simulator_config['categorical'].keys())}
    return {'plots': plots, 'metrics': metrics, 'info': info, 'session_id': sid}


# ── ROUTES ─────────────────────────────────────────────────────

@app.route('/web_plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('web_plots', filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        f = request.files.get('file')
        if not f:
            return jsonify({'error': 'No file uploaded'}), 400
        sid = str(uuid.uuid4())[:8]
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        # Overwrite the same file to prevent clutter
        path = os.path.join(upload_dir, "current_dataset.csv")
        f.save(path)
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        preview = df.head(5).to_dict(orient='records')
        dtypes = {c: str(df[c].dtype) for c in cols}
        uniques = {}
        for c in cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                uniques[c] = df[c].astype(str).unique().tolist()[:20]
        return jsonify({'session_id': sid, 'columns': cols, 'preview': preview,
                        'dtypes': dtypes, 'uniques': uniques, 'rows': len(df)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        sid = data['session_id']
        target_col = data['target_col']
        positive_val = data.get('positive_val', None)
        path = os.path.join("static", "uploads", "current_dataset.csv")
        df = pd.read_csv(path)
        result = run_pipeline(df, target_col, positive_val, sid)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        sid = data['session_id']
        player = data['player']
        if sid not in models_store:
            return jsonify({'error': 'Model not found for this session'}), 400
            
        store = models_store[sid]
        # Create 1-row DataFrame
        df_player = pd.DataFrame([player])
        # Add missing columns with defaults
        for c in store['raw_columns']:
            if c not in df_player.columns:
                df_player[c] = 0
                
        # Apply engineering with saved state
        X_eng, _ = apply_feature_engineering(df_player, store['state'])
        
        # Transform & Predict
        Xp = store['preprocessor'].transform(X_eng)
        proba = store['model'].predict_proba(Xp)[0][1]
        
        return jsonify({'churn_probability': float(proba)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

