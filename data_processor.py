import pandas as pd
import numpy as np
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. CORE PHYSICS ENGINE (Shared Logic)
# ==========================================
def apply_physics_engine(df):
    """
    The Single Source of Truth for Physics Calculations.
    Runs on BOTH training data and single test planets.
    """
    data = df.copy()
    
    # A. BASIC IMPUTATION (If columns exist, fill them; if not, create defaults)
    # This ensures your custom dict doesn't crash if you miss a field.
    defaults = {
        'pl_orbper': 365.25, 'pl_orbsmax': 1.0, 'pl_rade': 1.0, 'pl_radj': 0.1, 
        'pl_masse': 1.0, 'pl_massj': 0.003, 'pl_bmasse': 1.0, 'pl_dens': 5.51, 
        'pl_orbeccen': 0.0, 'pl_insol': 1.0, 'pl_eqt': 255.0, 'pl_ratdor': 215.0, 
        'pl_ratror': 0.009, 'st_teff': 5778.0, 'st_rad': 1.0, 'st_mass': 1.0, 
        'st_lum': 1.0, 'sy_pnum': 1
    }
    for col, val in defaults.items():
        if col not in data.columns:
            data[col] = val
        data[col] = data[col].fillna(val)

    # B. MASS & RADIUS STANDARDIZATION
    # Create _imputed columns if they don't exist
    if 'pl_masse_imputed' not in data.columns:
        # Priority: Earth Mass -> Jupiter Mass * Conv -> BMass -> Default
        data['pl_masse_imputed'] = data['pl_masse'].fillna(data['pl_massj'] * 317.8).fillna(data['pl_bmasse']).fillna(1.0)
    
    if 'pl_rade_imputed' not in data.columns:
        # Priority: Earth Rad -> Jupiter Rad * Conv -> Default
        data['pl_rade_imputed'] = data['pl_rade'].fillna(data['pl_radj'] * 11.2).fillna(1.0)

    # C. ADVANCED PHYSICS (The "Secret Sauce")
    # 1. Density Ratio (Earth = 1.0)
    data['density_ratio'] = data['pl_dens'] / 5.51
    
    # 2. Mass Ratio (Planet Mass / Star Mass)
    data['mass_ratio'] = data['pl_masse_imputed'] / (data['st_mass'] * 333000)
    
    # 3. Tidal Lock Proxy (Distance / Star Mass)
    data['tidal_lock_proxy'] = data['pl_ratdor'] / (data['st_mass'] + 1e-6)
    
    # 4. Thermal Contrast
    data['temp_diff_norm'] = (data['st_teff'] - data['pl_eqt']) / data['st_teff']
    
    # 5. Escape Velocity (v = sqrt(M/R))
    data['escape_vel'] = np.sqrt(data['pl_masse_imputed'] / (data['pl_rade_imputed'] + 1e-6))
    
    # 6. Retention Probability (Tanh activation)
    data['retention_prob'] = np.tanh(data['escape_vel'] * 3.0) 
    
    # 7. Stability Score (Gaussian Bell Curves for Temp & Insolation)
    t_score = np.exp(-((data['pl_eqt'] - 288)**2) / (2 * 50**2))
    i_score = np.exp(-((data['pl_insol'] - 1.0)**2) / (2 * 1.5**2))
    data['stability_score'] = (t_score + i_score) / 2

    return data

# ==========================================
# 2. DATA LOADING & TRAINING PREP
# ==========================================
def fetch_and_clean_data():
    """Fetches NASA data and applies the Physics Engine."""
    print("📡 [Data Processor] Fetching NASA Archive Data...")
    cols = [
        'pl_name', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 
        'pl_masse', 'pl_massj', 'pl_bmasse', 'pl_dens', 'pl_orbeccen', 
        'pl_insol', 'pl_eqt', 'pl_ratdor', 'pl_ratror', 'st_teff', 
        'st_rad', 'st_mass', 'st_lum', 'sy_pnum'
    ]
    try:
        raw_data = NasaExoplanetArchive.query_criteria(table="pscomppars", select=cols)
        df = raw_data.to_pandas()
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

    # Apply the Unified Physics Engine
    df = apply_physics_engine(df)
    
    # Planet Classification (Rocky vs Gas)
    df['planet_type'] = 0
    df.loc[df['pl_rade_imputed'] <= 1.75, 'planet_type'] = 1
    
    return df

def prepare_datasets(df_raw):
    """Prepares Training/Test splits with SMOTE."""
    df = df_raw.copy()
    
    # Teacher Labels (The Ground Truth for Training)
    def label_logic(row):
        if row['planet_type'] == 0: return 0
        if not (0.35 <= row['pl_insol'] <= 1.7): return 0
        if not (180 <= row['pl_eqt'] <= 310): return 0
        return 1
    
    df['habitable_candidate'] = df.apply(label_logic, axis=1)
    
    # STRICT Feature List (Must match exactly during inference)
    features = [
        'pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_ratdor', 'sy_pnum', 
        'tidal_lock_proxy', 'pl_dens', 'pl_insol', 'pl_eqt', 'pl_ratror', 
        'density_ratio', 'pl_masse_imputed', 'pl_rade_imputed', 'mass_ratio',
        'st_teff', 'st_rad', 'st_mass', 'st_lum', 'temp_diff_norm',
        'escape_vel', 'retention_prob', 'stability_score'
    ]
    
    X = df[features].fillna(0)
    y = df['habitable_candidate']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    print("🔄 [Data Processor] Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return {
        "X_train": X_train_res, "y_train": y_train_res,
        "X_test": X_test, "y_test": y_test,
        "full_df": df,
        "feature_order": features # Export list to ensure order safety
    }