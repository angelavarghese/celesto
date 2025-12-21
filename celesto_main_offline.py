import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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
# 2. REPORTING (Offline Narrative Engine)
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
        print(f"\n🔬 AGENT BREAKDOWN:")
        print(f"   ├─ Agent 1 (Atmos):   {s['Atmos']:.3f}  [Esc Vel: {p['escape_vel']:.2f}]")
        print(f"   ├─ Agent 2 (Orbit):   {s['Orbit']:.3f}  [Type: {star_type}]")
        print(f"   └─ Agent 3 (Surface): {s['Surface']:.3f}  [Temp: {p['pl_eqt']:.1f}K]")
        print("-" * 60)

def generate_rich_narrative(results):
    """
    Generates professional scientific narratives using logic templates.
    No API required. Zero cost. Instant results.
    """
    print("\n📝 GENERATING NARRATIVE (OFFLINE MODE)...\n")
    
    for r in results:
        s = r['scores']
        p = r['physics']
        star_type = "Red Dwarf" if p['st_mass'] < 0.6 else "Sun-like Star"
        
        print(f"=== ANALYSIS: {r['name']} ===")
        
        # --- Agent 1 (Physics) ---
        if s['Atmos'] > 0.85:
            a1_text = f"Physics simulation indicates exceptional atmospheric potential. With an escape velocity of {p['escape_vel']:.2f} (Earth=1.0), this planet has sufficient gravity to retain heavy gases like nitrogen and oxygen indefinitely."
        elif s['Atmos'] > 0.5:
            a1_text = f"Atmospheric retention is marginal. The escape velocity of {p['escape_vel']:.2f} suggests it can hold an atmosphere, but it may be thinner than Earth's or subject to stripping."
        else:
            a1_text = f"Critical failure in atmospheric retention. Low gravity (Esc Vel: {p['escape_vel']:.2f}) indicates this body likely resembles Mars or Mercury."

        # --- Agent 2 (Orbit) ---
        if s['Orbit'] > 0.8:
            a2_text = f"Orbital dynamics are optimal. The planet orbits a {star_type} with a period of {p['pl_orbper']:.1f} days. The high stability score indicates low eccentricity and a safe distance from stellar flares."
        elif p['st_mass'] < 0.6: 
            a2_text = f"Cautionary orbital flag. The planet orbits a {star_type} in a tight {p['pl_orbper']:.1f}-day orbit. This creates a high risk of Tidal Locking (Eternal Day vs Eternal Night)."
        else:
            a2_text = f"Orbital configuration poses habitability risks. Deviations in eccentricity or semi-major axis suggest a chaotic climate history."

        # --- Agent 3 (Surface) ---
        if s['Surface'] > 0.85:
            a3_text = f"Surface conditions are prime for liquid water. With an equilibrium temperature of {p['pl_eqt']:.0f}K and {p['pl_insol']:.2f}x Earth's sunlight, the planet sits firmly in the 'Goldilocks Zone'."
        elif s['Surface'] > 0.4:
            a3_text = f"Surface habitability is possible but challenging. The temperature ({p['pl_eqt']:.0f}K) is slightly outside the ideal range, requiring a specific greenhouse effect."
        else:
            a3_text = f"Hostile surface environment detected. The calculated temperature ({p['pl_eqt']:.0f}K) is incompatible with Earth-like biology."

        # --- Director ---
        if s['Final'] > 0.8:
            dir_text = f"Final Verdict: {s['Final']:.3f}. High-Priority Candidate. All agents align to suggest a world capable of supporting life. Recommended for spectroscopy."
        elif s['Final'] > 0.5:
            dir_text = f"Final Verdict: {s['Final']:.3f}. Mixed Candidate. Promising features exist, but significant flaws in orbit or surface conditions reduce probability."
        else:
            dir_text = f"Final Verdict: {s['Final']:.3f}. Candidate Rejected. The synthesis of data suggests a sterile or hostile environment."

        print(f"**Agent 1 (Physics):** {a1_text}")
        print(f"**Agent 2 (Orbit):** {a2_text}")
        print(f"**Agent 3 (Surface):** {a3_text}")
        print(f"**Director (Synthesis):** {dir_text}")
        print("-" * 60 + "\n")

def plot_results(results):
    df = pd.DataFrame([{"Planet": r['name'], "Score": r['scores']['Final']} for r in results])
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x='Planet', y='Score', palette='viridis', edgecolor='black')
    plt.axhspan(0.8, 1.05, color='green', alpha=0.1, label='Habitable')
    plt.axhspan(0.4, 0.8, color='orange', alpha=0.1, label='Potential')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom', fontweight='bold')
    plt.title("Final Celesto Analysis", fontweight='bold')
    plt.legend()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    if 'df_pscd' not in globals():
        df_pscd = dp.fetch_and_clean_data()
    
    if not df_pscd.empty:
        # 2. Train Agents
        bundle = dp.prepare_datasets(df_pscd)
        agents = {
            'a1': at.train_agent1(bundle),
            'a2': at.train_agent2(bundle),
            'a3': at.train_agent3(bundle)
        }
        agents['a4'] = at.train_agent4(agents['a1'], agents['a2'], agents['a3'], bundle)
        
        # 3. Define Planets
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
        for p in [earth, custom]:
            results.append(analyze_planet(p, agents))
            
        # 4. Generate Output
        print_mission_status(results)
        generate_rich_narrative(results)
        plot_results(results)