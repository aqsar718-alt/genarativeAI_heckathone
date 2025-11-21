# Exoplanet Habitability Predictor 🪐

A machine learning project to predict the habitability of exoplanets based on their physical characteristics. Built for a hackathon!

## 🚀 Quick Start

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    - Place your `exoplanets.xlsx` file in this directory.
    - It must contain columns: `radius`, `mass`, `orbit_distance`, `star_temperature`.
    - *Note: If no file is found, the script will generate a dummy dataset for demonstration.*

3.  **Train Model**:
    ```bash
    python train_model.py
    ```
    This will create `model_pipeline.pkl` and `evaluation_report.csv`.

4.  **Run App**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

- `train_model.py`: Main script for data processing, training, and evaluation.
- `app.py`: Streamlit web application for interactive predictions.
- `requirements.txt`: Python dependencies.
- `exoplanets.xlsx`: (User provided) Dataset file.

## 🧠 How it Works

1.  **Data Cleaning**: Removes rows with missing core values.
2.  **Labeling**: If no label exists, we use a conservative Habitable Zone (HZ) calculation based on star temperature and planet orbit, combined with mass/radius constraints.
3.  **Feature Engineering**: Calculates an **Earth Similarity Score (ESI)** based on radius, mass, and orbit.
4.  **Model**: Uses a **Random Forest Classifier** to predict habitability.

## ⚠️ Limitations & Ethics

- **Data Quality**: The model relies heavily on the accuracy of the input data. Exoplanet measurements often have high uncertainty.
- **Habitability Definition**: Our rule-based labeling is a simplification. Real habitability depends on atmosphere, magnetic field, water presence, etc., which are not included here.
- **Bias**: The model is biased towards "Earth-like" life as we know it.
- **Future Improvements**:
    - Incorporate atmospheric data if available.
    - Use more sophisticated HZ models (e.g., Kopparapu et al. 2013).
    - Add uncertainty quantification for predictions.

## 📝 Example Output

**Input**:
- Planet Name: Kepler-186f
- Radius: 1.1 Earths
- Mass: 1.1 Earths
- Orbit: 0.4 AU
- Star Temp: 3700 K

**Prediction**: 🌱 Habitable (Confidence: 85%)

**Explanation**:
- **Roman Urdu**: "Yeh planet habitable lagta hai kyun ke orbit distance habitable zone mein hai."
- **English**: "This planet seems habitable because orbit is within the habitable zone."
