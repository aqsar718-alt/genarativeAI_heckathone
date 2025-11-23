import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import google.generativeai as genai
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

# Gemini API Configuration
GENAI_API_KEY = "AIzaSyBLIikLFCKy0qR9G1AJd9A1Kjzmx6wWJU0"
genai.configure(api_key=GENAI_API_KEY)

def query_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

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
    Generates an explanation using Hugging Face API with a fallback to rule-based logic.
    """
    radius = inputs['radius']
    orbit = inputs['orbit_distance']
    temp = inputs['star_temperature']
    
    # 1. Prepare Rule-Based Fallback (in case API fails)
    # HZ Calculation
    lum = (temp / 5778)**4
    hz_in = 0.95 * np.sqrt(lum)
    hz_out = 1.35 * np.sqrt(lum)
    
    reasons_ur = []
    reasons_en = []
    
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
        
    if prediction == 1:
        intro_ur = "Khushkhabri! AI analysis ke mutabiq, yeh planet life support kar sakta hai."
        intro_en = "Good news! Based on AI analysis, this planet is a strong candidate for habitability."
        status = "Habitable"
    else:
        intro_ur = "Afsos, AI analysis batata hai ke yahan zindagi mushkil hai."
        intro_en = "Unfortunately, AI analysis suggests this planet is likely hostile to life."
        status = "Not Habitable"

    fallback_urdu = f"{intro_ur} {', aur '.join(reasons_ur)}."
    fallback_eng = f"{intro_en} Specifically, {', and '.join(reasons_en)}."

    # 2. Try Gemini API
    try:
        gravity = inputs['mass'] / (inputs['radius']**2)
        esi = inputs['esi']
        mass = inputs['mass']
        
        prompt = f"""
        [ROLE] You are a highly professional Astro-Analyst. Your task is to analyze the exoplanet data below and generate a comprehensive, structured mission report.

        [PLANET DATA]
        - Status: {status} (Confidence: {proba:.1%})
        - ESI Score: {esi:.1f}/100
        - Estimated Gravity: {gravity:.2f}g (Earth is 1.0g)
        - Planet Radius: {radius} Earth Radii
        - Planet Mass: {mass} Earth Masses
        - Orbit Distance: {orbit} AU
        - Star Temperature: {temp} K

        [OUTPUT STRUCTURE]
        Your entire response MUST strictly follow the three headings below. Do not add any extra text, introductory phrases, or conversation outside of these three sections.

        **1. NAME:** Generate a single, unique, and scientifically plausible designation name for this planet (e.g., 'Volcanus Prime', 'Terra Nova-6', or 'Icefang').
        **2. ENGLISH ANALYSIS:** Write a 4-sentence scientific analysis. 
            * **Sentence 1:** State the primary challenge or advantage (e.g., extremely high surface gravity, or perfectly within the Habitable Zone).
            * **Sentence 2:** Explain the ESI score and what it implies for habitability compared to Earth.
            * **Sentence 3:** Briefly discuss the impact of the Star Temperature and Orbit on potential liquid water.
            * **Sentence 4:** Suggest a preliminary scientific mission based on the data (e.g., 'Atmospheric composition probe recommended' or 'Long-duration orbital survey advised').
        **3. ROMAN URDU ANALYSIS:** Write a 4-sentence analysis explaining the same points in simple, clear Roman Urdu.
            * **Sentence 1:** Batao ke yahan zindagi ke liye **sabse bari rukawat ya faida** kya hai.
            * **Sentence 2:** ESI score ka matlab kya hai aur iski wajah se yeh Earth se kitna milta hai.
            * **Sentence 3:** Star ki garmi aur iske faasle (orbit) ka pani (liquid water) par kya asar hoga.
            * **Sentence 4:** Aage kya karna chahiye—ek **mission ka mashwara** do (jaise, 'Ek behtareen telescope se iski tasveer leni chahiye')
        """
        
        generated_text = query_gemini(prompt)
        
        if generated_text:
            # Robust parsing for the new format
            generated_name = "Unknown"
            eng_text = fallback_eng
            urdu_text = fallback_urdu
            
            try:
                # Split by the headers
                parts = generated_text.split("**1. NAME:**")
                if len(parts) > 1:
                    remaining = parts[1]
                    name_parts = remaining.split("**2. ENGLISH ANALYSIS:**")
                    if len(name_parts) > 1:
                        generated_name = name_parts[0].strip()
                        analysis_parts = name_parts[1].split("**3. ROMAN URDU ANALYSIS:**")
                        if len(analysis_parts) > 1:
                            eng_text = analysis_parts[0].strip()
                            urdu_text = analysis_parts[1].strip()
                            return status, urdu_text, eng_text, generated_name
            except:
                pass
                
            return status, urdu_text, eng_text, generated_name
                
    except Exception as e:
        pass
    
    # Return fallback if API fails
    return status, fallback_urdu, fallback_eng, "Unknown"

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
    st.markdown("### Generative Exoplanet Analysis System <span style='font-size: 0.8em; background: linear-gradient(90deg, #4285F4, #9B72CB); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold;'>⚡ Powered by Gemini</span>", unsafe_allow_html=True)
    
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
        defaults = example_planets.get(selected_preset)
        if defaults is None:
            defaults = {"radius": 1.0, "mass": 1.0, "orbit": 1.0, "temp": 5778}
        
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

    if analyze_btn:
        with st.spinner("Running Neural Net Simulations..."):
            time.sleep(1.5)
            
            # Calculations
            esi = calculate_esi(radius, mass, orbit)
            input_data = pd.DataFrame([[radius, mass, orbit, temp, esi]], 
                                    columns=['radius', 'mass', 'orbit_distance', 'star_temperature', 'earth_similarity_score'])
            
            prediction = pipeline.predict(input_data)[0]
            proba = pipeline.predict_proba(input_data)[0][1]
            
            status, urdu_exp, eng_exp, ai_name = generate_ai_explanation(
                {'radius': radius, 'orbit_distance': orbit, 'star_temperature': temp, 'mass': mass, 'esi': esi}, 
                prediction, proba
            )
            
            img_path = get_planet_image(prediction, temp, radius)
            
            # Store in Session State
            st.session_state['analysis_results'] = {
                'name': name,
                'ai_name': ai_name,
                'radius': radius,
                'mass': mass,
                'orbit': orbit,
                'temp': temp,
                'esi': esi,
                'prediction': prediction,
                'proba': proba,
                'status': status,
                'urdu_exp': urdu_exp,
                'eng_exp': eng_exp,
                'img_path': img_path
            }


    with col2:
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            # Results Display
            display_name = results['ai_name'] if results['ai_name'] != "Unknown" else results['name']
            st.markdown(f"### Analysis Result: {display_name}")
            if results['ai_name'] != "Unknown" and results['ai_name'] != results['name']:
                st.caption(f"User Designation: {results['name']}")
            
            # Dynamic Image
            if os.path.exists(results['img_path']):
                st.image(results['img_path'], caption="Generative Visualization of Surface Conditions", use_container_width=True)
            
            # Status Banner
            if results['prediction'] == 1:
                st.markdown(f"""
                <div class="success-box">
                    <h3>🌱 HABITABLE</h3>
                    <p>Confidence: {results['proba']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                if analyze_btn: # Only show balloons on fresh run
                    st.balloons()
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <h3>💀 NOT HABITABLE</h3>
                    <p>Confidence: {(1-results['proba']):.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("ESI Score", f"{results['esi']:.1f}", delta="Earth Similarity")
            m2.metric("Orbit Type", "Goldilocks" if "habitable zone" in results['eng_exp'] else "Extreme")
            m3.metric("Est. Gravity", f"{(results['mass']/results['radius']**2):.2f}g")

            # AI Explanation Tab
            st.markdown("#### 🧠 AI Mission Report")
            tab1, tab2 = st.tabs(["🇬🇧 English Analysis", "🇵🇰 Roman Urdu Analysis"])
            
            with tab1:
                st.markdown(f"""
                <div style="background: rgba(66, 133, 244, 0.1); border-left: 3px solid #4285F4; padding: 15px; border-radius: 5px;">
                    <strong style="color: #4285F4;">Gemini Analysis:</strong><br>
                    <span style="font-family: 'Courier New', monospace;">{results['eng_exp']}</span>
                </div>
                """, unsafe_allow_html=True)
            with tab2:
                st.markdown(f"""
                <div style="background: rgba(155, 114, 203, 0.1); border-left: 3px solid #9B72CB; padding: 15px; border-radius: 5px;">
                    <strong style="color: #9B72CB;">Roman Urdu Log:</strong><br>
                    <span style="font-family: 'Courier New', monospace;">{results['urdu_exp']}</span>
                </div>
                """, unsafe_allow_html=True)


                
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











