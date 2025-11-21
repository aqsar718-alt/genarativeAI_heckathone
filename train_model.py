import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

# Configuration
# Use the path provided by the user
DATA_DIR = r"c:\Users\aqsar\OneDrive\Desktop\aqsa_project\archive (3)"
DATA_FILE_CSV = os.path.join(DATA_DIR, "exoplanets.csv")
MODEL_FILE = "model_pipeline.pkl"
METADATA_FILE = "metadata.json"
EVAL_REPORT_FILE = "evaluation_report.csv"

def load_data():
    """Loads data from the CSV file and maps columns."""
    print(f"Loading data from {DATA_FILE_CSV}...")
    if not os.path.exists(DATA_FILE_CSV):
        raise FileNotFoundError(f"Could not find {DATA_FILE_CSV}")
    
    df = pd.read_csv(DATA_FILE_CSV)
    
    # Map columns
    # kepoi_name -> planet_name
    # koi_prad -> radius
    # koi_steff -> star_temperature
    # koi_period -> used to calculate orbit_distance
    # koi_disposition -> useful for filtering (CONFIRMED/CANDIDATE)
    
    column_mapping = {
        'kepoi_name': 'planet_name',
        'koi_prad': 'radius',
        'koi_steff': 'star_temperature',
        'koi_period': 'period_days'
    }
    
    # Rename columns that exist
    df = df.rename(columns=column_mapping)
    
    # Filter for relevant columns + helpers
    needed_cols = ['planet_name', 'radius', 'star_temperature', 'period_days', 'koi_disposition']
    # Check which ones are actually present
    present_cols = [c for c in needed_cols if c in df.columns]
    df = df[present_cols]
    
    return df

def calculate_orbit_distance(period_days):
    """
    Estimates orbit distance (semi-major axis) in AU using Kepler's 3rd Law.
    Assumption: Star mass is roughly 1 Solar Mass (simplification for hackathon).
    a (AU) = (P (years) ^ 2) ^ (1/3)
    """
    if pd.isna(period_days):
        return np.nan
    period_years = period_days / 365.25
    return (period_years ** 2) ** (1/3)

def estimate_mass(radius):
    """
    Estimates planet mass (Earth Masses) from radius (Earth Radii).
    Using a simplified power law: M ~ R^2.06 (Chen & Kipping 2017 approximation for smaller planets)
    """
    if pd.isna(radius):
        return np.nan
    return radius ** 2.06

def calculate_habitable_zone(star_temp):
    """
    Calculates conservative habitable zone bounds based on star temperature.
    Luminosity L approx (T / 5778)^4
    HZ inner = 0.95 * sqrt(L)
    HZ outer = 1.35 * sqrt(L)
    """
    if pd.isna(star_temp):
        return np.nan, np.nan
    luminosity = (star_temp / 5778.0) ** 4
    hz_inner = 0.95 * np.sqrt(luminosity)
    hz_outer = 1.35 * np.sqrt(luminosity)
    return hz_inner, hz_outer

def determine_habitability(row):
    """
    Rule-based habitability label.
    """
    if pd.isna(row['star_temperature']) or pd.isna(row['orbit_distance']) or pd.isna(row['radius']) or pd.isna(row['mass']):
        return 0
        
    hz_inner, hz_outer = calculate_habitable_zone(row['star_temperature'])
    
    in_hz = hz_inner <= row['orbit_distance'] <= hz_outer
    right_radius = 0.5 <= row['radius'] <= 2.5
    right_mass = 0.1 <= row['mass'] <= 10.0
    
    return 1 if (in_hz and right_radius and right_mass) else 0

def calculate_esi(row):
    """
    Simplified Earth Similarity Score (0-100).
    """
    if pd.isna(row['radius']) or pd.isna(row['mass']) or pd.isna(row['orbit_distance']):
        return 0
        
    r_e, m_e, d_e = 1.0, 1.0, 1.0
    w_r, w_m, w_d = 0.57, 1.07, 0.70
    
    diff_r = ((row['radius'] - r_e) / (row['radius'] + r_e)) ** 2
    diff_m = ((row['mass'] - m_e) / (row['mass'] + m_e)) ** 2
    diff_d = ((row['orbit_distance'] - d_e) / (row['orbit_distance'] + d_e)) ** 2
    
    weight_sum = w_r + w_m + w_d
    weighted_mean = (w_r * diff_r + w_m * diff_m + w_d * diff_d) / weight_sum
    
    esi = 1.0 - np.sqrt(weighted_mean)
    return max(0, esi * 100)

def preprocess(df):
    print("Preprocessing data...")
    
    # 1. Calculate missing features
    print("Calculating Orbit Distance...")
    df['orbit_distance'] = df['period_days'].apply(calculate_orbit_distance)
    
    print("Estimating Mass...")
    df['mass'] = df['radius'].apply(estimate_mass)
    
    # 2. Clean data
    required_cols = ['radius', 'mass', 'orbit_distance', 'star_temperature']
    
    # Drop rows with missing values in core features
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    print(f"Dropped {initial_len - len(df)} rows with missing values.")
    
    # 3. Generate Labels
    print("Generating 'habitable' labels...")
    df['habitable'] = df.apply(determine_habitability, axis=1)
    
    # 4. Feature Engineering: ESI
    print("Calculating ESI...")
    df['earth_similarity_score'] = df.apply(calculate_esi, axis=1)
    
    # Filter to reasonable physical values (remove outliers/errors in dataset)
    # e.g. radius > 100 Earth radii is likely an error or star
    df = df[df['radius'] < 30] 
    
    return df

def train_and_evaluate():
    df = load_data()
    df = preprocess(df)
    
    print(f"Final dataset size: {len(df)} rows")
    print(f"Habitable planets found: {df['habitable'].sum()}")
    
    features = ['radius', 'mass', 'orbit_distance', 'star_temperature', 'earth_similarity_score']
    target = 'habitable'
    
    X = df[features]
    y = df[target]
    
    # Handle class imbalance if needed (for now, standard split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    # Save Evaluation Report
    eval_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [acc, prec, rec, f1]
    })
    eval_df.to_csv(EVAL_REPORT_FILE, index=False)
    print(f"Evaluation report saved to {EVAL_REPORT_FILE}")
    
    # Save Model
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    # Save Metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'rows_used': len(df),
        'features': features,
        'test_accuracy': acc
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    train_and_evaluate()
