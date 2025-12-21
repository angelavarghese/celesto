import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import time

# Import your existing modules
import data_processor as dp
import agent_trainer as at

# ==========================================
# 1. APP CONFIG & CACHING
# ==========================================
st.set_page_config(page_title="Celesto Exoplanet Lab", page_icon="🪐", layout="wide")

@st.cache_data
def load_data():
    """Loads and cleans NASA data once."""
    return dp.fetch_and_clean_data()

@st.cache_resource
def train_models(_df):
    """Trains agents once."""
    bundle = dp.prepare_datasets(_df)
    agents = {
        'a1': at.train_agent1(bundle),
        'a2': at.train_agent2(bundle),
        'a3': at.train_agent3(bundle)
    }
    agents['a4'] = at.train_agent4(agents['a1'], agents['a2'], agents['a3'], bundle)
    return agents

def analyze_single_planet(planet_dict, agents):
    """Runs the full Celesto pipeline on a dictionary."""
    df_single = pd.DataFrame([planet_dict])
    df_eng = dp.apply_physics_engine(df_single).iloc[0]
    
    a1 = agents['a1']
    a1_in = a1['scaler'].transform(pd.DataFrame([df_eng[a1['features']]]))
    s1 = a1['model'].predict(a1_in)[0]
    
    a2 = agents['a2']
    a2_in = a2['scaler'].transform(pd.DataFrame([df_eng[a2['features']]]))
    s2 = a2['model'].predict_proba(a2_in)[0][1]
    
    a3 = agents['a3']
    a3_in = a3['scaler'].transform(pd.DataFrame([df_eng[a3['features']]]))
    s3 = a3['model'].predict_proba(a3_in)[0][1]
    
    a4 = agents['a4']
    raw_score = a4.predict_proba(np.array([[s1, s2, s3]]))[0][1]
    
    final = raw_score
    if "Earth" not in str(planet_dict.get('pl_name', '')):
        final = raw_score * 0.85
        
    return {
        "name": planet_dict.get('pl_name', 'Unknown'),
        "scores": {"Atmos": s1, "Orbit": s2, "Surface": s3, "Final": final},
        "physics": df_eng
    }

# ==========================================
# 2. UI LAYOUT
# ==========================================
st.title("🪐 Celesto: AI Habitability Lab")
st.markdown("Search the NASA Archive or **drag the sliders** to design a world and see if the AI thinks it's habitable.")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ Global Settings")
    use_api = st.checkbox("Use Gemini AI API", value=False)
    api_key = ""
    if use_api:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.info("💡 Tip: Try tweaking 'Distance' and 'Star Mass' to see how the Habitable Zone shifts!")

# --- LOAD SYSTEM ---
with st.spinner("🔭 Aligning Telescopes (Loading Data & Training Agents)..."):
    try:
        df_data = load_data()
        if df_data.empty:
            st.error("Failed to load NASA Data.")
            st.stop()
        agents = train_models(df_data)
        st.success("System Online.")
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# --- TABS: SEARCH VS CUSTOM ---
tab1, tab2 = st.tabs(["🔍 Search Database", "🛠️ Custom Builder"])

# ==========================================
# TAB 1: SEARCH DATABASE
# ==========================================
with tab1:
    st.subheader("Search the NASA Exoplanet Archive")
    
    # Dropdown with all planet names
    planet_list = sorted(df_data['pl_name'].unique())
    selected_planet_name = st.selectbox("Type or Select a Planet:", planet_list, index=planet_list.index("Kepler-186 f") if "Kepler-186 f" in planet_list else 0)
    
    if st.button("🚀 Analyze Selection", key="btn_search"):
        # 1. Retrieve Row
        planet_row = df_data[df_data['pl_name'] == selected_planet_name].iloc[0].to_dict()
        
        # 2. Analyze
        result = analyze_single_planet(planet_row, agents)
        
        # 3. Store for Display
        st.session_state['analysis_result'] = result

# ==========================================
# TAB 2: CUSTOM BUILDER (ALL SLIDERS)
# ==========================================
with tab2:
    st.subheader("Design a Synthetic World")
    
    p_name = st.text_input("Planet Name", "Pandora (Avatar)")
    
    # Organize sliders into logical columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌍 Planet Physics")
        p_mass = st.slider("Mass (Earths)", 0.1, 5.0, 0.8, 0.1, help="1.0 = Earth Mass")
        p_rad = st.slider("Radius (Earths)", 0.5, 2.5, 1.1, 0.1, help="1.0 = Earth Radius")
        p_dens = st.slider("Density (g/cm³)", 1.0, 8.0, 5.2, 0.1, help="Rock ~5.5, Water ~1.0")

    with col2:
        st.markdown("#### ☀️ Star System")
        s_mass = st.slider("Star Mass (Suns)", 0.08, 1.5, 0.12, 0.01, help="0.1 = Red Dwarf, 1.0 = Sun")
        s_temp = st.slider("Star Temp (K)", 2000, 7000, 3042, 50, help="Sun is 5778 K")
        p_sdist = st.slider("Distance (AU)", 0.01, 1.5, 0.05, 0.01, help="1.0 = Earth-Sun Distance")

    with col3:
        st.markdown("#### 🌡️ Environment")
        p_orb = st.slider("Period (Days)", 1.0, 400.0, 11.2, 0.1, help="Length of year")
        p_insol = st.slider("Insolation (Flux)", 0.1, 2.0, 0.65, 0.05, help="Sunlight intensity relative to Earth")
        # Estimate Temp for display guidance (not input)
        est_temp = int(s_temp * ((s_mass*0.9)/(2*p_sdist*215))**0.5 * 0.9)
        st.metric("Est. Surface Temp", f"{est_temp} K")

    # Derived for dict
    custom_planet = {
        'pl_name': p_name,
        'pl_orbper': p_orb, 'pl_orbsmax': p_sdist, 'pl_orbeccen': 0.0,
        'pl_rade_imputed': p_rad, 'pl_masse_imputed': p_mass, 'pl_dens': p_dens,
        'st_mass': s_mass, 'st_teff': s_temp,
        # We allow physics engine to calc real temp, or override with derived approximation
        'pl_eqt': est_temp, 
        'pl_insol': p_insol,
        'st_rad': s_mass * 0.9, 'st_lum': s_mass**3.5, 
        'pl_ratdor': (p_sdist * 215) / (s_mass * 0.9),
    }

    if st.button("🚀 Analyze Custom Build", key="btn_custom"):
        result = analyze_single_planet(custom_planet, agents)
        st.session_state['analysis_result'] = result

# ==========================================
# 3. RESULTS DASHBOARD (Shared)
# ==========================================
if 'analysis_result' in st.session_state:
    res = st.session_state['analysis_result']
    scores = res['scores']
    phys = res['physics']
    
    st.divider()
    st.header(f"Results for: {res['name']}")
    
    # 1. Top Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Habitability", f"{scores['Final']:.3f}", delta_color="normal")
    c2.metric("Atmos Score", f"{scores['Atmos']:.2f}")
    c3.metric("Orbit Score", f"{scores['Orbit']:.2f}")
    c4.metric("Surface Score", f"{scores['Surface']:.2f}")
    
    # 2. Detailed Breakdown
    st.subheader("🔬 Physics Telemetry")
    phy_col1, phy_col2 = st.columns(2)
    with phy_col1:
        st.write(f"**Star Type:** {'Red Dwarf' if phys['st_mass'] < 0.6 else 'Sun-like'}")
        st.write(f"**Escape Vel:** {phys['escape_vel']:.2f} (Earth=1.0)")
        st.write(f"**Tidal Lock Risk:** {'HIGH' if phys['tidal_lock_proxy'] > 400 else 'Low'}")
    with phy_col2:
        st.write(f"**Retention Prob:** {phys['retention_prob']:.2%}")
        st.write(f"**Stability Index:** {phys['stability_score']:.2f}")
        st.write(f"**Surface Temp:** {phys['pl_eqt']:.0f} K")
    
    # 3. Visualization
    st.subheader("📊 Agent Consensus")
    chart_data = pd.DataFrame({
        'Agent': ['Atmosphere', 'Orbit', 'Surface', 'FINAL'],
        'Score': [scores['Atmos'], scores['Orbit'], scores['Surface'], scores['Final']]
    })
    
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sns.barplot(data=chart_data, x='Score', y='Agent', palette='viridis', ax=ax)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.85, color='red', linestyle='--', label='Exoplanet Cap')
    st.pyplot(fig)
    
    # 4. Narrative
    st.subheader("📝 Mission Report")
    
    # Offline Generator Function
    def get_offline_text(s, p, name):
        txt = ""
        # Atmos
        if s['Atmos'] > 0.8: txt += f"**Agent 1 (Atmos):** Strong gravity ({p['escape_vel']:.2f}) suggests excellent atmospheric retention.\n\n"
        else: txt += f"**Agent 1 (Atmos):** Weak gravity ({p['escape_vel']:.2f}) indicates likely atmospheric stripping.\n\n"
        
        # Orbit
        if s['Orbit'] > 0.8: txt += f"**Agent 2 (Orbit):** Stable orbital configuration detected.\n\n"
        elif p['st_mass'] < 0.6: txt += f"**Agent 2 (Orbit):** Orbiting a Red Dwarf. Extreme Tidal Locking risk detected.\n\n"
        else: txt += f"**Agent 2 (Orbit):** Irregular orbital parameters suggest climate instability.\n\n"
        
        # Surface
        if s['Surface'] > 0.8: txt += f"**Agent 3 (Surface):** Conditions ({p['pl_eqt']:.0f}K) are ideal for liquid water.\n\n"
        else: txt += f"**Agent 3 (Surface):** Surface is likely too hostile for complex life.\n\n"
        
        # Director
        if s['Final'] > 0.8: txt += f"**Director Verdict:** High Priority Candidate. Recommended for spectroscopy."
        else: txt += f"**Director Verdict:** Candidate rejected based on current telemetry."
        
        return txt

    # API Logic
    if use_api and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Analyze planet: {res['name']}.
            Data: Mass {phys['pl_masse_imputed']:.2f}, Temp {phys['pl_eqt']:.0f}K, Star {phys['st_mass']:.2f} Suns.
            Scores: Atmos {scores['Atmos']:.2f}, Orbit {scores['Orbit']:.2f}, Surface {scores['Surface']:.2f}, Final {scores['Final']:.3f}.
            Write a 4-paragraph technical report.
            """
            with st.spinner("🤖 Contacting Gemini AI..."):
                response = model.generate_content(prompt)
                st.markdown(response.text)
        except Exception as e:
            st.warning(f"AI Error ({e}). Switching to Offline Mode.")
            st.markdown(get_offline_text(scores, phys, res['name']))
    else:
        st.info("Using Offline Narrative Engine (Enable API in sidebar for AI).")
        st.markdown(get_offline_text(scores, phys, res['name']))