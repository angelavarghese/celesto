import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import warnings
import time

# Custom Modules
import data_processor as dp
import agent_trainer as at

warnings.filterwarnings("ignore")

# ==========================================
# 1. ANALYSIS ENGINE
# ==========================================
def analyze_planet(planet_dict, agents):
    """
    Unified Inference Engine.
    Converts Dict -> DataFrame -> Physics -> Agents -> Detailed Report.
    """
    # 1. Standardize Data using the Unified Physics Engine
    df_single = pd.DataFrame([planet_dict])
    df_eng = dp.apply_physics_engine(df_single).iloc[0]
    
    # 2. Agent 1 (Atmosphere/Physics)
    a1 = agents['a1']
    a1_in = a1['scaler'].transform(pd.DataFrame([df_eng[a1['features']]]))
    s1 = a1['model'].predict(a1_in)[0]
    
    # 3. Agent 2 (Orbit)
    a2 = agents['a2']
    a2_in = a2['scaler'].transform(pd.DataFrame([df_eng[a2['features']]]))
    s2 = a2['model'].predict_proba(a2_in)[0][1]
    
    # 4. Agent 3 (Surface)
    a3 = agents['a3']
    a3_in = a3['scaler'].transform(pd.DataFrame([df_eng[a3['features']]]))
    s3 = a3['model'].predict_proba(a3_in)[0][1]
    
    # 5. Agent 4 (Synthesis)
    a4 = agents['a4']
    raw_score = a4.predict_proba(np.array([[s1, s2, s3]]))[0][1]
    
    # 6. Final Adjustments (STRICT 0.85 CAP)
    final = raw_score
    if "Earth" not in str(planet_dict.get('pl_name', '')):
        final = raw_score * 0.85
        
    return {
        "name": planet_dict.get('pl_name', 'Unknown'),
        "scores": {"Atmos": s1, "Orbit": s2, "Surface": s3, "Final": final},
        "physics": df_eng
    }

# ==========================================
# 2. REPORTING (Batch Mode)
# ==========================================
def print_mission_status(results):
    """Prints the detailed stats to the console."""
    for r in results:
        p = r['physics']
        s = r['scores']
        
        star_type = "Red Dwarf" if p['st_mass'] < 0.6 else "Sun-like"
        
        print(f"\n" + "="*60)
        print(f"🚀 REPORT: {r['name'].upper()}")
        print("="*60)
        
        print(f"\n📊 FINAL HABITABILITY SCORE: {s['Final']:.3f}")
        print(f"   (Raw Confidence: {s['Final']:.3f})")

        print(f"\n🔬 AGENT BREAKDOWN:")
        print(f"   ├─ Agent 1 (Atmos):   {s['Atmos']:.3f}  [Esc Vel: {p['escape_vel']:.2f} | Retention: {p['retention_prob']:.2f}]")
        print(f"   ├─ Agent 2 (Orbit):   {s['Orbit']:.3f}  [Type: {star_type} | Period: {p['pl_orbper']:.1f}d]")
        print(f"   └─ Agent 3 (Surface): {s['Surface']:.3f}  [Temp: {p['pl_eqt']:.1f}K | Flux: {p['pl_insol']:.2f}]")
        print("-" * 60)

def generate_gemini_report_batch(results, api_key):
    """
    BATCH NARRATIVE GENERATOR
    Sends ALL planets in ONE request to bypass Rate Limits.
    """
    if not api_key: return

    genai.configure(api_key=api_key)
    
    # Try preferred model first, then fallback
    # We avoid 2.0-flash-lite since it has strict quotas
    models_to_try = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.0-flash-lite']
    
    model = None
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(m)
            break
        except:
            continue
            
    if not model:
        print("❌ Could not connect to Gemini API. Using Offline Mode.")
        return

    print("\n📝 GENERATING BATCH NARRATIVE (1 Request)...")
    
    # Construct ONE giant prompt with all planets
    big_buffer = []
    for r in results:
        p = r['physics']
        s = r['scores']
        star_type = "Red Dwarf" if p['st_mass'] < 0.6 else "Sun-like"
        
        big_buffer.append(f"""
        === PLANET: {r['name']} ===
        TECHNICAL DATA:
        - Mass: {p['pl_masse_imputed']:.2f} Earths, Radius: {p['pl_rade_imputed']:.2f} Earths
        - Star: {star_type} ({p['st_mass']:.2f} Solar Mass)
        - Orbit: {p['pl_orbper']:.1f} days, Eccentricity: {p['pl_orbeccen']}
        - Surface: {p['pl_eqt']:.1f} K, Insolation: {p['pl_insol']:.2f} Earth Flux
        
        SCORES:
        - Agent 1 (Physics): {s['Atmos']:.2f}
        - Agent 2 (Orbit): {s['Orbit']:.2f}
        - Agent 3 (Surface): {s['Surface']:.2f}
        - FINAL VERDICT: {s['Final']:.3f}
        """)
        
    prompt = f"""
    Roleplay as 'Celesto', an AI Exoplanet Director. 
    Analyze the following list of planets based on the provided technical data and scores.
    
    DATA:
    {''.join(big_buffer)}
    
    INSTRUCTIONS:
    For EACH planet, write a report in this EXACT format:
    
    [PLANET NAME]
    **Agent 1 (Physics):** [Analysis of gravity/atmosphere]
    **Agent 2 (Orbit):** [Analysis of stability/year length]
    **Agent 3 (Surface):** [Analysis of water/temp]
    **Director (Synthesis):** [Final verdict]
    
    (Separate planets with a dashed line)
    """
    
    try:
        response = model.generate_content(prompt)
        print("\n" + response.text + "\n")
        print("-" * 60)
    except Exception as e:
        print(f"❌ API Error: {e}")
        # Fallback offline
        print("\n⚠️ API Failed. Showing Offline Summary.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    API_KEY = "AIzaSyBkhJCidSm_A_RA_NfxYNy55UBhW-9w0XM"
    
    # 1. Load & Train
    if 'df_pscd' not in globals():
        df_pscd = dp.fetch_and_clean_data()
    
    if not df_pscd.empty:
        bundle = dp.prepare_datasets(df_pscd)
        agents = {
            'a1': at.train_agent1(bundle),
            'a2': at.train_agent2(bundle),
            'a3': at.train_agent3(bundle)
        }
        agents['a4'] = at.train_agent4(agents['a1'], agents['a2'], agents['a3'], bundle)
        
        # 2. Define Planets
        earth = {
            'pl_name': 'Earth (Control)', 
            'pl_orbper': 365.25, 'pl_orbsmax': 1.0, 
            'pl_rade': 1.0, 'pl_masse': 1.0, 'pl_dens': 5.51, 
            'pl_insol': 1.0, 'pl_eqt': 255.0, 
            'st_teff': 5778.0, 'st_mass': 1.0, 'st_rad': 1.0
        }
        
        custom = {
            'pl_name': 'Pandora (Avatar)', 
            'pl_orbper': 11.2, 
            'pl_orbsmax': 0.05, 
            'pl_rade': 1.1, 
            'pl_masse': 0.8, 
            'pl_dens': 5.2, 
            'pl_insol': 0.65, 
            'pl_eqt': 234.0, 
            'st_teff': 3042.0, 
            'st_mass': 0.12, 
            'st_rad': 0.14
        }
        
        results = []
        # Process all planets locally first
        for p in [earth, custom]:
            results.append(analyze_planet(p, agents))
            
        # 3. Print Stats & Generate Batch Report
        print_mission_status(results)
        generate_gemini_report_batch(results, API_KEY)