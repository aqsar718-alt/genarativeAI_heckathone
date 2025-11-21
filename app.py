import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from PIL import Image

# Page Config
st.set_page_config(
    page_title="ExoHunter AI | Habitability Predictor",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Sci-Fi Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* General App Styling */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #000000 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    h1 { font-size: 3rem !important; }
    
    /* Cards & Containers */
    .css-1r6slb0, .stForm {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #005bea);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
    }
    
    /* Custom Classes */
    .highlight-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 10px;
    }
    .success-box {
        background: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
        padding: 15px;
        border-radius: 0 10px 10px 0;
    }
    .danger-box {
        background: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
        padding: 15px;
        border-radius: 0 10px 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_FILE = "model_pipeline.pkl"
METADATA_FILE = "metadata.json"
ASSETS_DIR = "assets"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return None

def calculate_esi(radius, mass, orbit_distance):
    r_e, m_e, d_e = 1.0, 1.0, 1.0
    w_r, w_m, w_d = 0.57, 1.07, 0.70
    
    diff_r = ((radius - r_e) / (radius + r_e)) ** 2
    diff_m = ((mass - m_e) / (mass + m_e)) ** 2
    diff_d = ((orbit_distance - d_e) / (orbit_distance + d_e)) ** 2
    
    weight_sum = w_r + w_m + w_d
    weighted_mean = (w_r * diff_r + w_m * diff_m + w_d * diff_d) / weight_sum
    
    esi = 1.0 - np.sqrt(weighted_mean)
    return max(0, esi * 100)

def get_planet_image(prediction, temp, radius):
    """Selects the appropriate image based on planet properties."""
    if prediction == 1:
        return os.path.join(ASSETS_DIR, "planet_habitable.png")
    
    # Logic for non-habitable visuals
    if radius > 6.0:
        return os.path.join(ASSETS_DIR, "planet_gas.png")
    if temp > 4000 or (temp > 3000 and radius < 2.0): # Rough heuristic
        return os.path.join(ASSETS_DIR, "planet_hot.png")
    if temp < 2000:
        return os.path.join(ASSETS_DIR, "planet_cold.png")
        
    return os.path.join(ASSETS_DIR, "planet_hot.png") # Default fallback

def generate_ai_explanation(inputs, prediction, proba):
    """
    Simulates a Generative AI response with scientific reasoning.
    """
    radius = inputs['radius']
    orbit = inputs['orbit_distance']
    temp = inputs['star_temperature']
    
    # HZ Calculation
    lum = (temp / 5778)**4
    hz_in = 0.95 * np.sqrt(lum)
    hz_out = 1.35 * np.sqrt(lum)
    
    reasons_ur = []
    reasons_en = []
    
    # Detailed Logic
    if orbit < hz_in:
        reasons_ur.append("orbit distance bohat kam hai, jis se satah ka darja hararat bohat ziyada hoga")
        reasons_en.append("the orbit is too close to the host star, likely resulting in surface temperatures too high for liquid water")
    elif orbit > hz_out:
        reasons_ur.append("orbit distance bohat ziyada hai, jis se pani jam jaye ga")
        reasons_en.append("the orbit is too far, suggesting a frozen world where water would exist only as ice")
    else:
        reasons_ur.append("orbit 'Goldilocks Zone' (habitable zone) mein hai jahan pani liquid form mein reh sakta hai")
        reasons_en.append("the orbit lies comfortably within the habitable zone, allowing for the potential existence of liquid water")
        
    if radius > 2.5:
        reasons_ur.append("radius Earth se bohat bara hai, shayed yeh gas giant hai")
        reasons_en.append("the radius indicates a massive planet, likely a gas giant with no solid surface")
    elif radius < 0.8:
        reasons_ur.append("radius Earth se chota hai, atmosphere shayad weak ho")
        reasons_en.append("the radius is smaller than Earth's, which might imply a thin atmosphere unable to retain heat")
        
    # Constructing the narrative
    if prediction == 1:
        intro_ur = "Khushkhabri! AI analysis ke mutabiq, yeh planet life support kar sakta hai."
        intro_en = "Good news! Based on AI analysis, this planet is a strong candidate for habitability."
        status = "Habitable"
    else:
        intro_ur = "Afsos, AI analysis batata hai ke yahan zindagi mushkil hai."
        intro_en = "Unfortunately, AI analysis suggests this planet is likely hostile to life."
        status = "Not Habitable"

    urdu_text = f"{intro_ur} {', aur '.join(reasons_ur)}."
    eng_text = f"{intro_en} Specifically, {', and '.join(reasons_en)}."
    
    return status, urdu_text, eng_text

def main():
    # Sidebar - Mission Control
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3212/3212567.png", width=100)
        st.title("Mission Control")
        st.markdown("---")
        
        st.subheader("📡 Model Status")
        metadata = load_metadata()
        if metadata:
            st.success("Online")
            st.caption(f"Accuracy: {metadata.get('test_accuracy', 0):.2%}")
            st.caption(f"Last Training: {metadata.get('training_date', '').split('T')[0]}")
        else:
            st.error("Offline")
            
        st.markdown("---")
        st.markdown("### 🛠️ Settings")
        show_metrics = st.checkbox("Show Telemetry Data", value=False)
        
    # Main Content
    st.title("🌌 ExoHunter AI")
    st.markdown("### Generative Exoplanet Analysis System")
    
    pipeline = load_model()
    if not pipeline:
        st.error("⚠️ Model Pipeline Missing. Please initialize via `train_model.py`.")
        return

    # Layout: 2 Columns (Input Panel | Visualization Panel)
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("#### 🔭 Planetary Parameters")
        
        # Preset Loader
        example_planets = {
            "Select a Preset...": None,
            "Earth 2.0 (Kepler-186f)": {"radius": 1.1, "mass": 1.1, "orbit": 0.4, "temp": 3700},
            "Super Earth (Kepler-452b)": {"radius": 1.6, "mass": 5.0, "orbit": 1.05, "temp": 5757},
            "Lava World (55 Cancri e)": {"radius": 1.9, "mass": 8.0, "orbit": 0.015, "temp": 5196},
            "Ice Giant (Kepler-16b)": {"radius": 8.5, "mass": 105.0, "orbit": 0.7, "temp": 4500}
        }
        
        selected_preset = st.selectbox("", list(example_planets.keys()), label_visibility="collapsed")
        defaults = example_planets.get(selected_preset, {"radius": 1.0, "mass": 1.0, "orbit": 1.0, "temp": 5778})
        
        with st.form("analysis_form"):
            name = st.text_input("Planet Designation", value="Proxima Centauri b")
            
            c1, c2 = st.columns(2)
            with c1:
                radius = st.number_input("Radius (R⊕)", value=float(defaults['radius']), step=0.1)
                mass = st.number_input("Mass (M⊕)", value=float(defaults['mass']), step=0.1)
            with c2:
                orbit = st.number_input("Orbit (AU)", value=float(defaults['orbit']), step=0.01)
                temp = st.number_input("Star Temp (K)", value=float(defaults['temp']), step=50.0)
                
            analyze_btn = st.form_submit_button("🚀 Initiate Analysis")

    with col2:
        if analyze_btn:
            # Processing Animation
            with st.spinner("Running Neural Net Simulations..."):
                time.sleep(1.5) # Dramatic pause for effect
                
                # Calculations
                esi = calculate_esi(radius, mass, orbit)
                input_data = pd.DataFrame([[radius, mass, orbit, temp, esi]], 
                                        columns=['radius', 'mass', 'orbit_distance', 'star_temperature', 'earth_similarity_score'])
                
                prediction = pipeline.predict(input_data)[0]
                proba = pipeline.predict_proba(input_data)[0][1]
                
                status, urdu_exp, eng_exp = generate_ai_explanation(
                    {'radius': radius, 'orbit_distance': orbit, 'star_temperature': temp}, 
                    prediction, proba
                )
                
                img_path = get_planet_image(prediction, temp, radius)

            # Results Display
            st.markdown(f"### Analysis Result: {name}")
            
            # Dynamic Image
            if os.path.exists(img_path):
                st.image(img_path, caption="Generative Visualization of Surface Conditions", use_container_width=True)
            
            # Status Banner
            if prediction == 1:
                st.markdown(f"""
                <div class="success-box">
                    <h3>🌱 HABITABLE</h3>
                    <p>Confidence: {proba:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <h3>💀 NOT HABITABLE</h3>
                    <p>Confidence: {(1-proba):.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("ESI Score", f"{esi:.1f}", delta="Earth Similarity")
            m2.metric("Orbit Type", "Goldilocks" if "habitable zone" in eng_exp else "Extreme")
            m3.metric("Est. Gravity", f"{(mass/radius**2):.2f}g")

            # AI Explanation Tab
            st.markdown("#### 🧠 AI Mission Report")
            tab1, tab2 = st.tabs(["🇬🇧 English Analysis", "🇵🇰 Roman Urdu Analysis"])
            
            with tab1:
                st.markdown(f"**Mission Log:** {eng_exp}")
            with tab2:
                st.markdown(f"**Mission Log:** {urdu_exp}")
                
        else:
            # Idle State
            st.info("👈 Enter planetary data to begin simulation.")
            st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop", 
                     caption="Awaiting Input...", use_container_width=True)

    # Telemetry Section (Optional)
    if show_metrics and os.path.exists("evaluation_report.csv"):
        st.markdown("---")
        st.subheader("📊 System Telemetry")
        df_eval = pd.read_csv("evaluation_report.csv")
        st.dataframe(df_eval, use_container_width=True)

if __name__ == "__main__":
    main()
