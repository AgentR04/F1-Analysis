"""
F1 Championship Predictor - Model Export Utility
Export trained models for Streamlit application
"""

import pickle
import joblib
import pandas as pd
import numpy as np
import os

def create_sample_data_for_demo():
    """
    Create sample data for Streamlit demo when models aren't available
    """
    print("🏎️ Creating sample demo data...")

    # Sample F1 data for demonstration
    drivers = ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Sergio Pérez',
              'George Russell', 'Carlos Sainz', 'Lando Norris', 'Fernando Alonso',
              'Oscar Piastri', 'Alexander Albon']

    teams = ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Red Bull Racing', 'Mercedes',
            'Ferrari', 'McLaren', 'Aston Martin', 'McLaren', 'Williams']

    # Generate realistic predictions based on current F1 standings
    sample_data = pd.DataFrame({
        'Driver': drivers,
        'Team': teams,
        'Predicted_Points': [575, 240, 285, 195, 165, 155, 130, 115, 85, 25],
        'Win_Probability': [0.85, 0.35, 0.45, 0.25, 0.15, 0.20, 0.18, 0.08, 0.05, 0.01],
        'Championship_Odds': [1.2, 8.5, 4.2, 15.0, 35.0, 25.0, 40.0, 120.0, 200.0, 1000.0],
        'Season': [2024] * 10
    })

    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Save sample data
    sample_data.to_pickle('data/sample_predictions.pkl')

    print("✅ Sample demo data created!")
    print("- data/sample_predictions.pkl")

if __name__ == "__main__":
    create_sample_data_for_demo()