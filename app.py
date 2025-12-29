import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Custom modules (Ensure data_processor.py and agent_trainer.py are in your directory)
import data_processor as dp
import agent_trainer as at

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================

def get_offline_text(s, p, name):
    """Rule-based narrative engine for offline use or API failure."""
    return f"""
    ### 🛡️ Offline Report: {name}
    * **Agent 1 (Atmos):** {'High' if s['Atmos'] > 0.8 else 'Low'} retention potential (Esc Vel: {p['escape_vel']:.2f}x Earth).
    * **Agent 2 (Orbit):** {'Stable' if s['Orbit'] > 0.8 else 'Instability/Tidal Risk'} detected for current orbital period.
    * **Agent 3 (Surface):** {'Optimal' if s['Surface'] > 0.8 else 'Extreme'} thermal environment ({p['pl_eqt']:.0f}K).
    * **Verdict:** {'High Priority' if s['Final'] > 0.75 else 'Candidate Rejected'} for spectroscopic follow-up.
    """

@st.cache_data
def load_data():
    return dp.fetch_and_clean_data()

@st.cache_resource
def train_models(_df):
    bundle = dp.prepare_datasets(_df)
    agents = {
        'a1': at.train_agent1(bundle),
        'a2': at.train_agent2(bundle),
        'a3': at.train_agent3(bundle)
    }
    agents['a4'] = at.train_agent4(agents['a1'], agents['a2'], agents['a3'], bundle)
    return agents

def analyze_single_planet(planet_dict, agents):
    df_single = pd.DataFrame([planet_dict])
    df_eng = dp.apply_physics_engine(df_single).iloc[0]
    
    # Inference
    s1 = agents['a1']['model'].predict(agents['a1']['scaler'].transform(pd.DataFrame([df_eng[agents['a1']['features']]])))[0]
    s2 = agents['a2']['model'].predict_proba(agents['a2']['scaler'].transform(pd.DataFrame([df_eng[agents['a2']['features']]])))[0][1]
    s3 = agents['a3']['model'].predict_proba(agents['a3']['scaler'].transform(pd.DataFrame([df_eng[agents['a3']['features']]])))[0][1]
    
    # Director Synthesis
    raw_score = agents['a4'].predict_proba(np.array([[s1, s2, s3]]))[0][1]
    final = raw_score if "Earth" in str(planet_dict.get('pl_name', '')) else raw_score * 0.85
        
    return {"name": planet_dict.get('pl_name', 'Unknown'), "scores": {"Atmos": s1, "Orbit": s2, "Surface": s3, "Final": final}, "physics": df_eng}

# ==========================================
# 2. APP CONFIG & UI (Toggles Preserved)
# ==========================================
st.set_page_config(page_title="Celesto Exoplanet Lab", page_icon="🪐", layout="wide")

if 'narrative_cache' not in st.session_state:
    st.session_state['narrative_cache'] = ""

st.title("🪐 Celesto: AI Habitability Lab")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ Global Settings")
    use_api = st.checkbox("Use GenAI API", value=False)
    api_key = ""
    if use_api:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.info("💡 Tip: Try tweaking 'Distance' and 'Star Mass' to see how the Habitable Zone shifts!")

# --- LOAD SYSTEM ---
with st.spinner("🔭 Initializing Multi-Agent System..."):
    df_data = load_data()
    agents = train_models(df_data)

# --- TABS: SEARCH VS CUSTOM ---
tab1, tab2 = st.tabs(["🔍 Search Database", "🛠️ Custom Builder"])

# ==========================================
# TAB 1: SEARCH DATABASE
# ==========================================
with tab1:
    st.subheader("Search the NASA Exoplanet Archive")
    planet_list = sorted(df_data['pl_name'].unique())
    selected_planet_name = st.selectbox("Type or Select a Planet:", planet_list)
    
    if st.button("🚀 Analyze Selection", key="btn_search"):
        planet_row = df_data[df_data['pl_name'] == selected_planet_name].iloc[0].to_dict()
        st.session_state['analysis_result'] = analyze_single_planet(planet_row, agents)
        st.session_state['narrative_cache'] = ""

# ==========================================
# TAB 2: CUSTOM BUILDER (Sliding Toggles)
# ==========================================
with tab2:
    st.subheader("Design a Synthetic World")
    p_name = st.text_input("Planet Name", "Pandora")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🌍 Planet Physics")
        p_mass = st.slider("Mass (Earths)", 0.1, 5.0, 1.0)
        p_rad = st.slider("Radius (Earths)", 0.5, 2.5, 1.0)
    with col2:
        st.markdown("#### ☀️ Star System")
        s_mass = st.slider("Star Mass (Suns)", 0.08, 1.5, 1.0)
        s_temp = st.slider("Star Temp (K)", 2000, 7000, 5778)
    with col3:
        st.markdown("#### 🌡️ Environment")
        p_sdist = st.slider("Distance (AU)", 0.01, 2.0, 1.0)
        p_orb = st.slider("Period (Days)", 1.0, 500.0, 365.0)

    if st.button("🚀 Analyze Custom Build", key="btn_custom"):
        custom = {
            'pl_name': p_name, 'pl_orbper': p_orb, 'pl_orbsmax': p_sdist,
            'pl_rade_imputed': p_rad, 'pl_masse_imputed': p_mass, 'pl_dens': 5.5,
            'st_mass': s_mass, 'st_teff': s_temp, 'pl_eqt': 255/(p_sdist**0.5),
            'pl_insol': 1/(p_sdist**2), 'st_rad': s_mass, 'st_lum': s_mass**3.5,
            'pl_ratdor': 215*p_sdist, 'pl_orbeccen': 0.0
        }
        st.session_state['analysis_result'] = analyze_single_planet(custom, agents)
        st.session_state['narrative_cache'] = ""

# ==========================================
# 3. RESULTS DASHBOARD
# ==========================================
if 'analysis_result' in st.session_state:
    res = st.session_state['analysis_result']
    s, p = res['scores'], res['physics']
    
    st.divider()
    st.header(f"Results for: {res['name']}")
    
    # 1. Top Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Habitability", f"{s['Final']:.3f}")
    c2.metric("Atmos Score", f"{s['Atmos']:.2f}")
    c3.metric("Orbit Score", f"{s['Orbit']:.2f}")
    c4.metric("Surface Score", f"{s['Surface']:.2f}")

    

    # 2. Concise Mission Report
    st.subheader("📝 Mission Report")
    if use_api and api_key:
        if st.button("✨ Generate AI Analysis"):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"""
                Analyze {res['name']}. Provide a CONCISE bulleted report corresponding to these scores:
                - Agent 1 (Atmos): {s['Atmos']:.2f} (Mass/Gravity retention)
                - Agent 2 (Orbit): {s['Orbit']:.2f} (Stellar stability)
                - Agent 3 (Surface): {s['Surface']:.2f} (Thermal environment)
                - Director Verdict: {s['Final']:.2f}
                Use professional, telegraphic language. Max 2 short sentences per bullet.
                """
                with st.spinner("🤖 Consulting Director..."):
                    response = model.generate_content(prompt)
                    st.session_state['narrative_cache'] = response.text
            except Exception as e:
                st.session_state['narrative_cache'] = get_offline_text(s, p, res['name'])
        
        if st.session_state['narrative_cache']:
            st.markdown(st.session_state['narrative_cache'])
    else:
        st.markdown(get_offline_text(s, p, res['name']))