#!/usr/bin/env python3
"""
Clinical Crystal Ball: Quick Demo Script
Generate sample data and launch demo dashboard for immediate testing
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def create_demo_data():
    """Create minimal demo data for immediate dashboard testing"""

    print("Creating Clinical Crystal Ball Demo Data...")

    # Create sample static patient data
    np.random.seed(42)
    n_patients = 50

    static_data = []
    for pid in range(1, n_patients + 1):
        static_data.append({
            'patient_id': pid,
            'age': round(np.random.normal(65, 10), 1),
            'gender': np.random.choice(['Male', 'Female']),
            'condition_type': np.random.choice(['Diabetes', 'Heart_Failure', 'Both'], p=[0.5, 0.3, 0.2]),
            'baseline_bmi': round(np.random.normal(28, 4), 1)
        })

    static_df = pd.DataFrame(static_data)
    static_df.to_csv('patient_static_data.csv', index=False)

    # Create sample time series data (30 days per patient)
    time_series_data = []
    n_days = 30

    for patient in static_data:
        pid = patient['patient_id']

        for day in range(n_days):
            # Generate realistic vitals with some variation
            base_glucose = 130 if patient['condition_type'] == 'Diabetes' else 110
            glucose = np.clip(np.random.normal(base_glucose, 25), 70, 300)

            bp_sys = np.clip(np.random.normal(130, 20), 90, 180)
            hr = np.clip(np.random.normal(75, 10), 50, 120)
            weight = np.random.normal(75, 15)

            time_series_data.append({
                'patient_id': pid,
                'date': (datetime(2024, 1, 1) + timedelta(days=day)).strftime('%Y-%m-%d'),
                'day_index': day,
                'blood_glucose': round(glucose, 1),
                'blood_pressure_systolic': round(bp_sys, 1),
                'heart_rate': round(hr, 1),
                'weight': round(weight, 1),
                'bmi': round(weight / (1.7**2), 1),
                'medication_adherence': round(np.random.beta(7, 2), 2),
                'daily_steps': round(max(0, np.random.normal(5000, 2000))),
                'sleep_hours': round(np.clip(np.random.normal(7, 1.5), 4, 11), 1),
                'cholesterol_hdl': round(np.random.normal(45, 10), 1),
                'cholesterol_ldl': round(np.random.normal(120, 25), 1),
                'creatinine': round(np.random.normal(1.1, 0.3), 2),
                'hba1c': round(np.random.normal(7, 1), 1),
                'deterioration_event': np.random.binomial(1, 0.02),
                'deterioration_in_next_90d': np.random.binomial(1, 0.15)
            })

    ts_df = pd.DataFrame(time_series_data)
    ts_df.to_csv('synthetic_patient_data.csv', index=False)
    ts_df.to_csv('processed_patient_data.csv', index=False)  # Simplified for demo

    print(f"Created demo dataset: {n_patients} patients, {len(ts_df)} records")

    # Create sample model predictions
    predictions_data = []
    for pid in range(1, n_patients + 1):
        risk_score = np.random.beta(2, 5)  # Most patients low risk

        # Make some patients high risk
        if np.random.random() < 0.1:  # 10% high risk
            risk_score = np.random.uniform(0.7, 0.95)

        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'High'
        elif risk_score > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        predictions_data.append({
            'patient_id': pid,
            'split': 'test',
            'y_true': np.random.binomial(1, risk_score),
            'risk_score': round(risk_score, 3),
            'prediction': int(risk_score > 0.5),
            'risk_level': risk_level
        })

    pred_df = pd.DataFrame(predictions_data)
    pred_df.to_csv('model_predictions.csv', index=False)

    # Create sample evaluation results
    evaluation_results = {
        'performance_metrics': {
            'Model Performance Metrics': {
                'AUC-ROC': '0.8523',
                'AUC-PRC': '0.4205',
                'Sensitivity (Recall)': '0.7891',
                'Specificity': '0.8456',
                'Positive Predictive Value': '0.3421',
                'Negative Predictive Value': '0.9234',
                'F1-Score': '0.4756'
            }
        },
        'risk_stratification': [
            {'Risk Level': 'High', 'Count': 8, 'Actual Positive Rate': '0.625', 'Average Risk Score': '0.812', 'Percentage of Total': '16.0%'},
            {'Risk Level': 'Medium', 'Count': 15, 'Actual Positive Rate': '0.267', 'Average Risk Score': '0.485', 'Percentage of Total': '30.0%'},
            {'Risk Level': 'Low', 'Count': 27, 'Actual Positive Rate': '0.074', 'Average Risk Score': '0.156', 'Percentage of Total': '54.0%'}
        ],
        'patient_explanations': [
            {
                'patient_id': 5,
                'risk_score': '0.842',
                'actual_outcome': 1,
                'top_risk_drivers': [
                    'Elevated glucose trend over 30 days',
                    'Decreased medication adherence',
                    'Multiple comorbidities present'
                ],
                'recommended_actions': [
                    'Schedule immediate physician consultation',
                    'Review and adjust medication regimen',
                    'Implement enhanced monitoring protocol'
                ]
            }
        ],
        'global_feature_importance': {
            'feature_0': 0.15,
            'feature_1': 0.12,
            'feature_2': 0.10,
            'feature_3': 0.08,
            'feature_4': 0.07
        }
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    # Create minimal model artifact
    import joblib

    model_artifacts = {
        'model': None,  # Placeholder for demo
        'performance_metrics': evaluation_results['performance_metrics']['Model Performance Metrics'],
        'feature_importance': evaluation_results['global_feature_importance']
    }

    joblib.dump(model_artifacts, 'tft_model.pkl')

    print("Demo data generation complete!")
    print(f"Files created:")
    print(f"   - patient_static_data.csv ({len(static_df)} patients)")
    print(f"   - synthetic_patient_data.csv ({len(ts_df)} records)")  
    print(f"   - model_predictions.csv ({len(pred_df)} predictions)")
    print(f"   - evaluation_results.json (performance metrics)")
    print(f"   - tft_model.pkl (model artifacts)")

    return True


def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\nLaunching Clinical Crystal Ball Dashboard...")
    print("Dashboard will open in your browser at: http://localhost:8501")
    print("Loading dashboard components...")

    try:
        import subprocess
        import sys

        # Launch streamlit dashboard
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            '05_streamlit_dashboard.py',
            '--server.headless', 'false',
            '--server.runOnSave', 'true'
        ])

    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("Try running manually: streamlit run 05_streamlit_dashboard.py")


def main():
    """Main demo execution"""
    print("Clinical Crystal Ball - Quick Demo")
    print("=" * 50)
    print("This demo creates sample data and launches the dashboard")
    print("Perfect for exploring the system before running full pipeline")
    print("=" * 50)

    # Check if streamlit is available
    try:
        import streamlit
        print("Streamlit available")
    except ImportError:
        print("Streamlit not found. Please install: pip install streamlit")
        return False

    # Create demo data
    success = create_demo_data()

    if not success:
        print("Demo data creation failed")
        return False

    print("\nDemo Setup Complete!")
    print("Ready to launch dashboard...")

    # Prompt user
    response = input("\nLaunch dashboard now? (y/n): ").lower().strip()

    if response in ['y', 'yes']:
        launch_dashboard()
    else:
        print("\nTo launch dashboard later, run:")
        print("   streamlit run 05_streamlit_dashboard.py")

    print("\nDemo files are ready for exploration!")
    print("Check the generated CSV files to explore the data structure")

    return True


if __name__ == "__main__":
    main()
