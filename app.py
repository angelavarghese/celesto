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
        api_key = st.text_input("GenAI API Key", type="password")
    
    st.info("💡 Tip: Try tweaking 'Distance' and 'Star Mass' to see how the Habitable Zone shifts!")

# --- LOAD SYSTEM ---
with st.spinner("🔭 Initializing Multi-Agent System..."):
    df_data = load_data()
    agents = train_models(df_data)

# --- TABS: SEARCH VS CUSTOM ---
# tab1, tab2 = st.tabs(["🔍 Search Database", "🛠️ Custom Builder"])
tab1, tab2, tab3 = st.tabs(["🔍 Search Database", "🛠️ Custom Builder", "📊 Model Validation"])

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
            # Core Sliders
# --- User Input (Mapped to Sliders) ---
            'pl_name': p_name, 
            'pl_orbper': p_orb, 
            'pl_orbsmax': p_sdist,
            'pl_rade_imputed': p_rad, 
            'pl_masse': p_mass, 
            'st_mass': s_mass, 
            'st_teff': s_temp, 
            
            # --- Direct Features (Required by ML Index) ---
            'pl_dens': 5.51,        # Earth density default
            'pl_orbeccen': 0.0,     # Circular orbit default
            'pl_ratdor': 215 * p_sdist, # Distance/Radius ratio
            'pl_ratror': 0.009,     # Planet/Star radius ratio
            'sy_pnum': 1,           # Number of planets
            
            # --- Physics Placeholders (Calculated by apply_physics_engine) ---
            'pl_eqt': np.nan, 
            'pl_insol': np.nan, 
            'st_rad': s_mass,       # Proxy: R scales with M for main sequence
            'st_lum': np.nan,
            'density_ratio': np.nan,
            'mass_ratio': np.nan,
            'tidal_lock_proxy': np.nan,
            'temp_diff_norm': np.nan,
            'escape_vel': np.nan,
            'retention_prob': np.nan,
            'stability_score': np.nan,

            # --- Structural Keys (Required for data_processor logic) ---
            'pl_massj': np.nan, 
            'pl_radj': np.nan, 
            'pl_bmasse': np.nan, 
            'pl_rade': np.nan,
            'st_spectype': None
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


# ... (Tab 1 and Tab 2 code remains the same) ...

# ==========================================
# TAB 3: MODEL VALIDATION (Ablation & Metrics)
# ==========================================
with tab3:
    st.header("🔬 Scientific Validation Dashboard")
    st.info("This tab displays the performance metrics and ROC/PR curves used in the IEEE paper.")

    if st.button("📈 Run Performance Suite"):
        # 1. Prepare the test data from the bundle
        # Note: You'll need to expose 'bundle' from your train_models function
        bundle = dp.prepare_datasets(df_data)
        X_test = bundle['X_test']
        y_test = bundle['y_test']

        # 2. Generate Predictions for all agents
        # Atmosphere (Agent 1)
        p1 = agents['a1']['model'].predict(agents['a1']['scaler'].transform(X_test[agents['a1']['features']].fillna(0)))
        
        # Orbit & Surface (Agents 2 & 3)
        p2 = agents['a2']['model'].predict_proba(agents['a2']['scaler'].transform(X_test[agents['a2']['features']].fillna(0)))[:, 1]
        p3 = agents['a3']['model'].predict_proba(agents['a3']['scaler'].transform(X_test[agents['a3']['features']].fillna(0)))[:, 1]
        
        # Director (Agent 4)
        X_meta_test = np.column_stack((p1, p2, p3))
        final_probs = agents['a4'].predict_proba(X_meta_test)[:, 1]
        final_preds = (final_probs > 0.5).astype(int)

        # 3. Call your reporting functions (from agent_trainer.py)
        st.subheader("Final System Performance")
        metrics = at.report_metrics(y_test, final_preds, final_probs, "Celesto Director")
        
        # Display metrics as Streamlit Columns
        col1, col2, col3 = st.columns(3)
        col1.metric("F1 Score", f"{metrics['F1']:.4f}")
        col2.metric("Accuracy", f"{metrics['Acc']:.4f}")
        col3.metric("Recall", f"{metrics['Rec']:.4f}")

        # 4. Show the ROC/PR Curves
        st.subheader("ROC and Precision-Recall Curves")
        # We wrap the plt.show() logic to work with Streamlit
        fig = at.plot_curves(y_test, {
            "Atmos Agent": p1, 
            "Orbit Agent": p2, 
            "Surface Agent": p3, 
            "Full Director": final_probs
        })
        st.pyplot(plt.gcf()) # Display the matplotlib figure in Streamlit
# --- TABS: SEARCH VS CUSTOM VS VALIDATION ---
