"""
F1 Championship Predictor - Model Export Utility
Export trained models for Streamlit application
"""

import pickle
import joblib
import pandas as pd
import numpy as np

def export_models_for_streamlit():
    """
    Export trained models and preprocessed data for Streamlit app
    Run this after training all models in the Jupyter notebook
    """
    
    print(" F1 MODEL EXPORT UTILITY")
print("=" * 40)

# Note: These variables should be available from your Jupyter notebook session
# If running separately, you'll need to train the models first

try:
# Export the best performing model (Neural Network)
print(" Exporting Neural Network model...")
joblib.dump(best_predictor, 'models/nn_model.pkl')
joblib.dump(best_scaler, 'models/nn_scaler.pkl')

# Export other models for comparison
print(" Exporting XGBoost model...")
joblib.dump(xgb_model, 'models/xgb_model.pkl')

print(" Exporting Random Forest model...")
joblib.dump(rf_model, 'models/rf_model.pkl')

print(" Exporting SVR model...")
joblib.dump(svr_model, 'models/svr_model.pkl')
joblib.dump(scaler_svr, 'models/svr_scaler.pkl')

# Export preprocessed data
print(" Exporting processed data...")
season_enhanced.to_pickle('data/season_enhanced.pkl')
ml_data.to_pickle('data/ml_data.pkl')

# Export feature information
feature_info = {
'enhanced_features': enhanced_features,
'season_features': season_features,
'model_performance': {
'nn_r2': nn_r2,
'nn_mae': nn_mae,
'xgb_r2': xgb_r2,
'xgb_mae': xgb_mae,
'rf_r2': rf_r2,
'rf_mae': rf_mae
}
}

with open('data/feature_info.pkl', 'wb') as f:
pickle.dump(feature_info, f)

print(" All models and data exported successfully!")
print("\nFiles created:")
print("- models/nn_model.pkl (Neural Network)")
print("- models/nn_scaler.pkl (Neural Network Scaler)")
print("- models/xgb_model.pkl (XGBoost)")
print("- models/rf_model.pkl (Random Forest)")
print("- models/svr_model.pkl (SVR)")
print("- models/svr_scaler.pkl (SVR Scaler)")
print("- data/season_enhanced.pkl (Processed Data)")
print("- data/ml_data.pkl (ML Dataset)")
print("- data/feature_info.pkl (Feature Information)")

except NameError as e:
print(f" Error: {e}")
print("Please run this after executing all model training cells in the Jupyter notebook")
print("Or ensure all required variables are available in the current session")

def create_sample_data_for_demo():
    """
    Create sample data for Streamlit demo when models aren't available
    """
    print(" Creating sample demo data...")

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
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Save sample data
    sample_data.to_pickle('data/sample_predictions.pkl')

    print(" Sample demo data created!")
    print("- data/sample_predictions.pkl")

if __name__ == "__main__":
import os

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Choose an option:")
print("1. Export models from Jupyter notebook session")
print("2. Create sample demo data")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
export_models_for_streamlit()
elif choice == "2":
create_sample_data_for_demo()
else:
print("Invalid choice. Creating sample demo data...")
create_sample_data_for_demo()