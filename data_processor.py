import pandas as pd
import numpy as np
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. CORE PHYSICS ENGINE (Expanded & Tracked)
# ==========================================
def apply_physics_engine(df):
    data = df.copy()
    TOTAL_ROWS = len(data)

    # --- TRACKING: BEFORE ---
    before_teff = data['st_teff'].isna().sum()
    before_rad = data['st_rad'].isna().sum()
    before_mass = data['st_mass'].isna().sum()
    before_orbs = data['pl_orbsmax'].isna().sum()
    before_eqt = data['pl_eqt'].isna().sum()
    before_lum = data['st_lum'].isna().sum()
    before_insol = data['pl_insol'].isna().sum()

    print(f"\n📡 [Physics Engine] Initial Null Values:")
    print(f"   Stellar: Teff={before_teff}, Rad={before_rad}, Mass={before_mass}, Lum={before_lum}")
    print(f"   Orbital: Orbsmax={before_orbs}, Eqt={before_eqt}, Insol={before_insol}")

    # --- STEP A: STELLAR PARAMETER INFERENCE ---
    med_teff = data['st_teff'].median()
    med_rad = data['st_rad'].median()
    med_mass = data['st_mass'].median()

    def calculate_st_teff(row):
        if pd.notna(row['st_teff']) and row['st_teff'] != 0: return row['st_teff']
        teff_ranges = {'O':(30000,52000), 'B':(10000,30000), 'A':(7500,10000), 'F':(6000,7500), 'G':(5200,6000), 'K':(3700,5200), 'M':(2400,3700)}
        spectype = str(row.get('st_spectype', ''))
        if spectype and spectype[0] in teff_ranges:
            mi, ma = teff_ranges[spectype[0]]
            sub = int(spectype[1]) if len(spectype) > 1 and spectype[1].isdigit() else 5
            return ma - (sub/9)*(ma-mi)
        if pd.notna(row['st_lum']) and row['st_lum'] > 0 and pd.notna(row['st_rad']) and row['st_rad'] > 0:
            return (row['st_lum'] / (4 * np.pi * (row['st_rad'] ** 2) * 5.67e-8)) ** 0.25
        return med_teff

    def calculate_st_rad(row):
        if pd.notna(row['st_rad']) and row['st_rad'] != 0: return row['st_rad']
        rad_ranges = {'O':(6.6,20), 'B':(1.8,6.6), 'A':(1.4,1.8), 'F':(1.15,1.4), 'G':(0.96,1.15), 'K':(0.7,0.96), 'M':(0.1,0.7)}
        spectype = str(row.get('st_spectype', ''))
        if spectype and spectype[0] in rad_ranges:
            return sum(rad_ranges[spectype[0]]) / 2
        if pd.notna(row['st_lum']) and row['st_lum'] > 0 and pd.notna(row['st_teff']) and row['st_teff'] > 0:
            # Use max(0, val) to prevent negative roots
            return np.sqrt(max(0, row['st_lum'] / (4 * np.pi * 5.67e-8 * (row['st_teff'] ** 4))))
        return med_rad

    def calculate_st_mass(row):
        if pd.notna(row['st_mass']) and row['st_mass'] != 0: return row['st_mass']
        mass_ranges = {'O':(16,100), 'B':(2.1,16), 'A':(1.4,2.1), 'F':(1.04,1.4), 'G':(0.8,1.04), 'K':(0.45,0.8), 'M':(0.08,0.45)}
        spectype = str(row.get('st_spectype', ''))
        if spectype and spectype[0] in mass_ranges:
            mi, ma = mass_ranges[spectype[0]]
            sub = int(spectype[1]) if len(spectype) > 1 and spectype[1].isdigit() else 5
            return ma - (sub/9)*(ma-mi)
        return med_mass

    data['st_teff'] = data.apply(calculate_st_teff, axis=1)
    data['st_rad'] = data.apply(calculate_st_rad, axis=1)
    data['st_mass'] = data.apply(calculate_st_mass, axis=1)

    # --- STEP B: ORBITAL & THERMAL VECTORIZED PHYSICS ---
    # 1. Kepler's 3rd Law
    mask_a = data['pl_orbsmax'].isna() & data['pl_orbper'].notna()
    safe_mass = np.maximum(1e-6, data.loc[mask_a, 'st_mass'])
    data.loc[mask_a, 'pl_orbsmax'] = ((data.loc[mask_a, 'pl_orbper']/365.25)**2 * safe_mass)**(1/3)

    # 2. Stefan-Boltzmann for Luminosity
    mask_l = data['st_lum'].isna()
    data.loc[mask_l, 'st_lum'] = (data.loc[mask_l, 'st_rad']**2) * (data.loc[mask_l, 'st_teff']/5778)**4

    # 3. Equilibrium Temperature (T_eq)
    AU_TO_SOLAR_RADIUS = 214.935
    ALBEDO = 0.3
    mask_eq = data['pl_eqt'].isna() & data['pl_orbsmax'].notna()
    # Safe distance to prevent divide by zero or complex roots
    safe_a = np.maximum(1e-6, data.loc[mask_eq, 'pl_orbsmax'] * AU_TO_SOLAR_RADIUS)
    data.loc[mask_eq, 'pl_eqt'] = data.loc[mask_eq, 'st_teff'] * np.sqrt(data.loc[mask_eq, 'st_rad'] / (2 * safe_a)) * (1 - ALBEDO)**(1/4)

    # 4. Insolation
    mask_i = data['pl_insol'].isna() & (data['pl_orbsmax'] > 0)
    data.loc[mask_i, 'pl_insol'] = data.loc[mask_i, 'st_lum'] / (data.loc[mask_i, 'pl_orbsmax'] ** 2)

    # --- FINAL SAFETY: Ensure Real Numbers ---
    # This fixes the AttributeError by using numpy's real function on the series values
    data['pl_orbsmax'] = np.real(data['pl_orbsmax'].values)
    data['pl_eqt'] = np.real(data['pl_eqt'].values)
    data['pl_insol'] = np.real(data['pl_insol'].values)

    # --- STEP C: MASS/RADIUS STANDARDIZATION ---
    data['pl_masse_imputed'] = data['pl_masse'].fillna(data['pl_massj'] * 317.8).fillna(data['pl_bmasse']).fillna(1.0)
    data['pl_rade_imputed'] = data['pl_rade'].fillna(data['pl_radj'] * 11.2).fillna(1.0)

    data = data.fillna(0)

    # --- STEP D: FEATURE ENGINEERING ---
    data['density_ratio'] = data['pl_dens'].fillna(5.51) / 5.51
    data['mass_ratio'] = data['pl_masse_imputed'] / (data['st_mass'] * 333000)
    data['tidal_lock_proxy'] = data['pl_ratdor'].fillna(215) / (data['st_mass'] + 1e-6)
    data['temp_diff_norm'] = (data['st_teff'] - data['pl_eqt'].fillna(255)) / data['st_teff']
    data['escape_vel'] = np.sqrt(np.maximum(0, data['pl_masse_imputed'] / (data['pl_rade_imputed'] + 1e-6)))
    data['retention_prob'] = np.tanh(data['escape_vel'] * 3.0)
    data['stability_score'] = (np.exp(-((data['pl_eqt'].fillna(255)-288)**2)/(2*50**2)) + np.exp(-((data['pl_insol'].fillna(1.0)-1.0)**2)/(2*1.5**2))) / 2
    
    data['pl_orbeccen'] = data['pl_orbeccen'].fillna(data['pl_orbeccen'].median()).clip(0, 0.999)

    # --- STEP E: PLANET TYPE CLASSIFICATION ---
    data['planet_type'] = 0
    data.loc[(data['pl_rade_imputed'] <= 1.75) & (data['pl_dens'].fillna(5.51) >= 2.0), 'planet_type'] = 1

    # --- TRACKING: AFTER ---
    print(f"✅ [Physics Engine] Imputation Complete:")
    print(f"   Stellar: Teff filled: {before_teff - data['st_teff'].isna().sum()} | Rad filled: {before_rad - data['st_rad'].isna().sum()}")
    print(f"   Orbital: Orbsmax filled: {before_orbs - data['pl_orbsmax'].isna().sum()} | Eqt filled: {before_eqt - data['pl_eqt'].isna().sum()}")

    return data

def fetch_and_clean_data():
    print("📡 [Data Processor] Fetching NASA Archive Data...")
    cols = ['pl_name', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_masse', 'pl_massj', 
            'pl_bmasse', 'pl_dens', 'pl_orbeccen', 'pl_insol', 'pl_eqt', 'pl_ratdor', 'pl_ratror', 
            'st_teff', 'st_rad', 'st_mass', 'st_lum', 'sy_pnum', 'st_spectype']
    try:
        raw_data = NasaExoplanetArchive.query_criteria(table="pscomppars", select=cols)
        df = raw_data.to_pandas()
    except Exception as e:
        print(f"❌ Error: {e}"); return pd.DataFrame()

    df = apply_physics_engine(df)
    return df

def prepare_datasets(df_raw):
    df = df_raw.copy()

    # Teacher Label Logic
    def label_logic(row):
        if row['planet_type'] == 0: return 0
        if not (0.35 <= row['pl_insol'] <= 1.7): return 0
        if not (180 <= row['pl_eqt'] <= 310): return 0
        return 1

    df['habitable_candidate'] = df.apply(label_logic, axis=1)
    
    features = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_ratdor', 'sy_pnum', 'tidal_lock_proxy', 
                'pl_dens', 'pl_insol', 'pl_eqt', 'pl_ratror', 'density_ratio', 'pl_masse_imputed', 
                'pl_rade_imputed', 'mass_ratio', 'st_teff', 'st_rad', 'st_mass', 'st_lum', 
                'temp_diff_norm', 'escape_vel', 'retention_prob', 'stability_score']

    # Final Strict Check before ML
    for col in features:
        null_count = df[col].isna().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"🧹 [Safety Impute] '{col}': Filled {null_count} remaining nulls with median.")

    X = df[features]
    y = df['habitable_candidate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
# Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"🔄 [Data Processor] Label Counts - Habitable: {y_train.sum()} | Hostile: {len(y_train) - y_train.sum()}")

    # --- SAFE SMOTE IMPLEMENTATION ---
    X_train_res, y_train_res = X_train, y_train  # Default fallback if SMOTE fails
    
    if y_train.sum() > 1: # SMOTE needs at least 2 samples of the minority class to work
        try:
            print("🔄 [Data Processor] Balancing classes with SMOTE...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"✅ [Data Processor] SMOTE Complete. New Train Size: {len(X_train_res)}")
        except Exception as e:
            print(f"⚠️ [Data Processor] SMOTE skipped: {e}. Proceeding with imbalanced data.")
    else:
        print("⚠️ [Data Processor] Insufficient Habitable samples for SMOTE. Using raw distribution.")

    # Final check to ensure variables exist before returning
    return {
        "X_train": X_train_res, 
        "y_train": y_train_res, 
        "X_test": X_test, 
        "y_test": y_test, 
        "full_df": df, 
        "features": features
    }