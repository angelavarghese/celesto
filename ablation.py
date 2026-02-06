import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score

# Custom modules
import data_processor as dp
import agent_trainer as at

def run_ablation():
    print("🚀 [Ablation] Starting Study...")
    
    # 1. Load data
    df_raw = dp.fetch_and_clean_data()
    bundle = dp.prepare_datasets(df_raw)
    
    X_train = bundle['X_train']
    y_train = bundle['y_train']
    X_test = bundle['X_test']
    y_test = bundle['y_test']

    # ---------------------------------------------------------
    # CONFIG A: SINGLE-AGENT BASELINE (SCALED)
    # ---------------------------------------------------------
    print("📉 Evaluating Baseline: Scaled Single Logistic Regression...")
    # make_pipeline ensures scaling is applied to test data correctly
    baseline = make_pipeline(
        StandardScaler(), 
        LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    )
    baseline.fit(X_train, y_train)
    b_preds = baseline.predict(X_test)
    
    # ---------------------------------------------------------
    # CONFIG B: CELESTO STACKED ENSEMBLE
    # ---------------------------------------------------------
    print("🏗️ Evaluating Celesto: Multi-Agent Stacked Architecture...")
    
    # Train base experts
    a1 = at.train_agent1(bundle)
    a2 = at.train_agent2(bundle)
    a3 = at.train_agent3(bundle)
    
    # Train Director using the NEW leakage-free logic
    director = at.train_agent4(a1, a2, a3, bundle)
    
    # Final Test Set Inference
    p1 = a1['model'].predict(a1['scaler'].transform(X_test[a1['features']].fillna(0)))
    p2 = a2['model'].predict_proba(a2['scaler'].transform(X_test[a2['features']].fillna(0)))[:, 1]
    p3 = a3['model'].predict_proba(a3['scaler'].transform(X_test[a3['features']].fillna(0)))[:, 1]
    
    X_meta_test = np.column_stack((p1, p2, p3))
    # ensemble_preds = director.predict(X_meta_test)
    probs = director.predict_proba(X_meta_test)[:, 1]
    ensemble_preds = (probs > 0.8).astype(int) # Only flag if 80% sure

    # ---------------------------------------------------------
    # 3. FINAL RESULTS
    # ---------------------------------------------------------
    results = {
        "Baseline": {
            "F1": f1_score(y_test, b_preds),
            "Accuracy": accuracy_score(y_test, b_preds),
            "Recall": recall_score(y_test, b_preds)
        },
        "Celesto MAS": {
            "F1": f1_score(y_test, ensemble_preds),
            "Accuracy": accuracy_score(y_test, ensemble_preds),
            "Recall": recall_score(y_test, ensemble_preds)
        }
    }

    print("\n" + "="*40)
    print(f"{'METRIC':<15} | {'BASELINE':<10} | {'CELESTO MAS':<10}")
    print("-" * 40)
    for m in ["F1", "Accuracy", "Recall"]:
        print(f"{m:<15} | {results['Baseline'][m]:<10.4f} | {results['Celesto MAS'][m]:<10.4f}")
    print("="*40)

if __name__ == "__main__":
    run_ablation()