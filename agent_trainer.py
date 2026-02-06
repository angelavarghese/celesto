import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_agent1(data_bundle):
    print("☁️ [Trainer] Training Agent 1 (Atmosphere)...")
    df = data_bundle['full_df']
    target = (0.5 * df['retention_prob']) + (0.5 * df['stability_score'])
    features = ['pl_eqt', 'pl_insol', 'pl_dens', 'escape_vel', 'retention_prob', 'stability_score', 'st_teff', 'st_rad']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_scaled, target)
    return {"model": model, "scaler": scaler, "features": features}

def train_agent2(data_bundle):
    print("🪐 [Trainer] Training Agent 2 (Orbit)...")
    features = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_ratdor', 'sy_pnum', 'st_mass', 'tidal_lock_proxy', 'mass_ratio']
    X_train = data_bundle['X_train'][features]
    y_train = data_bundle['y_train']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y_train)
    return {"model": model, "scaler": scaler, "features": features}

def train_agent3(data_bundle):
    print("🌡️ [Trainer] Training Agent 3 (Surface)...")
    features = ['pl_insol', 'pl_eqt', 'pl_dens', 'density_ratio', 'pl_ratror', 'st_teff', 'st_rad', 'st_lum', 'temp_diff_norm']
    X_train = data_bundle['X_train'][features]
    y_train = data_bundle['y_train']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    base_svm = SVC(kernel='rbf', C=0.8, probability=True, random_state=42)
    model = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)
    model.fit(X_scaled, y_train)
    return {"model": model, "scaler": scaler, "features": features}

def train_agent4(a1, a2, a3, data_bundle):
    print("👑 [Trainer] Training Agent 4 (Synthesis)...")
    X_test = data_bundle['X_test']
    y_test = data_bundle['y_test']
    
    # Generate Predictions
    a1_in = a1['scaler'].transform(X_test[a1['features']].fillna(0))
    p1 = a1['model'].predict(a1_in)
    
    a2_in = a2['scaler'].transform(X_test[a2['features']].fillna(0))
    p2 = a2['model'].predict_proba(a2_in)[:, 1]
    
    a3_in = a3['scaler'].transform(X_test[a3['features']].fillna(0))
    p3 = a3['model'].predict_proba(a3_in)[:, 1]
    
    X_meta = np.column_stack((p1, p2, p3))
    model = LogisticRegression(class_weight='balanced', random_state=42)
    # model = LogisticRegression(random_state=42)
    model.fit(X_meta, y_test)
    
    print(f"   Trust Weights -> Atmos: {model.coef_[0][0]:.2f} | Orbit: {model.coef_[0][1]:.2f} | Surface: {model.coef_[0][2]:.2f}")
    return model


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, mean_absolute_error, mean_squared_error, 
                             r2_score, roc_curve, precision_recall_curve, auc)

def report_metrics(y_true, y_pred, y_prob=None, model_name="Model", is_regression=False):
    print(f"\n📊 --- {model_name} Metrics ---")
    if is_regression:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
    else:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"CM: {model_name}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        return {"Acc": acc, "F1": f1, "Rec": rec}

def plot_curves(y_true, probs_dict):
    """Generates ROC and PR curves for all agents and the final Director."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, probs in probs_dict.items():
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ax2.plot(recall, precision, label=f'{name}')
    
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend()
    ax2.set_title('Precision-Recall (PR) Curve')
    ax2.legend()
    plt.tight_layout()
    plt.show()