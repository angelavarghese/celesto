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
    model.fit(X_meta, y_test)
    
    print(f"   Trust Weights -> Atmos: {model.coef_[0][0]:.2f} | Orbit: {model.coef_[0][1]:.2f} | Surface: {model.coef_[0][2]:.2f}")
    return model