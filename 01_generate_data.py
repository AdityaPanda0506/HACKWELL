# Phase 1: Generate High-Quality Synthetic Clinical Dataset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_synthetic_data():
    """
    Generate realistic synthetic dataset for chronic care patients 
    with time-series data for predicting deterioration in next 90 days
    """

    NUM_PATIENTS = 500
    NUM_DAYS = 90

    print(f"Generating {NUM_PATIENTS} patients x {NUM_DAYS} days...")

    # Generate static patient data
    static_data = []
    for pid in range(1, NUM_PATIENTS + 1):
        age = np.clip(np.random.normal(65, 12), 30, 90)
        gender = np.random.choice(['Male', 'Female'])
        condition = np.random.choice(['Diabetes', 'Heart_Failure', 'Both'], p=[0.45, 0.35, 0.2])
        bmi = np.clip(np.random.normal(28, 5), 18, 45)

        static_data.append({
            'patient_id': pid,
            'age': round(age, 1),
            'gender': gender,
            'condition_type': condition,
            'baseline_bmi': round(bmi, 1)
        })

    # Generate time series data
    time_series_data = []

    for patient in static_data:
        pid = patient['patient_id']
        condition = patient['condition_type']

        # Set condition-specific parameters
        if condition == 'Diabetes':
            glucose_base = np.random.normal(140, 20)
            bp_base = np.random.normal(135, 15)
            risk = 0.12
        elif condition == 'Heart_Failure':
            glucose_base = np.random.normal(110, 15)
            bp_base = np.random.normal(125, 20)
            risk = 0.15
        else:  # Both conditions
            glucose_base = np.random.normal(150, 25)
            bp_base = np.random.normal(145, 20)
            risk = 0.18

        # Determine deterioration events
        will_deteriorate = np.random.random() < risk
        deterioration_day = np.random.randint(20, NUM_DAYS-20) if will_deteriorate else -1

        # Generate daily data
        for day in range(NUM_DAYS):
            # Calculate deterioration effect
            det_effect = 1.0
            if deterioration_day > 0:
                days_to_event = deterioration_day - day
                if 0 >= days_to_event >= -5:  # Event and post-event
                    det_effect = 1.4
                elif 10 >= days_to_event > 0:  # Pre-event buildup
                    det_effect = 1.1 + (10 - days_to_event) * 0.03

            # Generate vitals with realistic correlations
            glucose = glucose_base * det_effect + np.random.normal(0, 15)
            glucose = np.clip(glucose, 70, 400)

            bp_sys = bp_base * det_effect + np.random.normal(0, 12)
            bp_sys = np.clip(bp_sys, 90, 200)

            hr = 75 + (det_effect - 1) * 30 + np.random.normal(0, 8)
            hr = np.clip(hr, 50, 130)

            weight = patient['baseline_bmi'] * 1.75**2 + np.random.normal(0, 1)
            current_bmi = weight / (1.75**2)

            # Medication adherence (worse during deterioration periods)
            if det_effect > 1.2:
                adherence = np.random.beta(3, 4)  # Poor adherence
            else:
                adherence = np.random.beta(8, 2)  # Good adherence

            # Lifestyle factors
            steps = max(0, np.random.normal(5000, 2000) * (2 - det_effect))
            sleep = np.clip(np.random.normal(7, 1.5) * (2 - det_effect * 0.5), 3, 12)

            # Lab values (updated every ~30 days)
            if day % 30 == 0 or day == 0:
                hdl = np.random.normal(45, 10) * (1.1 if patient['gender'] == 'Female' else 1.0)
                ldl = np.random.normal(120, 25) * det_effect
                creatinine = np.random.normal(1.1, 0.2) * det_effect

                if condition in ['Diabetes', 'Both']:
                    hba1c = 6.5 + (glucose_base - 100) * 0.02 * det_effect + np.random.normal(0, 0.5)
                else:
                    hba1c = np.random.normal(5.5, 0.3)

            # Add missing data patterns
            if np.random.random() < 0.08:
                steps = np.nan
            if np.random.random() < 0.06:
                sleep = np.nan

            time_series_data.append({
                'patient_id': pid,
                'date': (datetime(2024, 1, 1) + timedelta(days=day)).strftime('%Y-%m-%d'),
                'day_index': day,
                'blood_glucose': round(glucose, 1),
                'blood_pressure_systolic': round(bp_sys, 1),
                'heart_rate': round(hr, 1),
                'weight': round(weight, 1),
                'bmi': round(current_bmi, 1),
                'medication_adherence': round(adherence, 2),
                'daily_steps': round(steps) if not pd.isna(steps) else np.nan,
                'sleep_hours': round(sleep, 1) if not pd.isna(sleep) else np.nan,
                'cholesterol_hdl': round(hdl, 1),
                'cholesterol_ldl': round(ldl, 1),
                'creatinine': round(creatinine, 2),
                'hba1c': round(hba1c, 1),
                'deterioration_event': int(day == deterioration_day) if deterioration_day > 0 else 0
            })

    # Convert to DataFrames
    static_df = pd.DataFrame(static_data)
    ts_df = pd.DataFrame(time_series_data)

    # Generate deterioration_in_next_90d labels
    print("Computing deterioration labels...")
    ts_df['deterioration_in_next_90d'] = 0

    for pid in range(1, NUM_PATIENTS + 1):
        if pid % 100 == 0:
            print(f"   Processing patient {pid}/{NUM_PATIENTS}")

        patient_data = ts_df[ts_df['patient_id'] == pid].copy()

        if patient_data['deterioration_event'].sum() > 0:
            event_day = patient_data[patient_data['deterioration_event'] == 1]['day_index'].iloc[0]
            # Mark 30 days before event as positive (reduced window for efficiency)
            mask = (patient_data['day_index'] >= max(0, event_day - 30)) & (patient_data['day_index'] < event_day)
            ts_df.loc[ts_df['patient_id'] == pid, 'deterioration_in_next_90d'] = mask.astype(int)

    return static_df, ts_df

if __name__ == "__main__":
    # Generate the synthetic data
    static_df, ts_df = generate_synthetic_data()

    # Save to CSV files
    static_df.to_csv('patient_static_data.csv', index=False)
    ts_df.to_csv('synthetic_patient_data.csv', index=False)

    # Print summary statistics
    print(f"\nDataset Generation Complete!")
    print(f"Static data: {len(static_df)} patients")
    print(f"Time series data: {len(ts_df)} records")
    print(f"Deterioration events: {ts_df['deterioration_event'].sum()}")
    print(f"Positive labels: {ts_df['deterioration_in_next_90d'].sum()} ({ts_df['deterioration_in_next_90d'].mean():.1%})")

    print(f"\nCondition distribution:")
    print(static_df['condition_type'].value_counts())

    print(f"\nSample time series data:")
    print(ts_df[['patient_id', 'day_index', 'blood_glucose', 'blood_pressure_systolic', 
                 'deterioration_event', 'deterioration_in_next_90d']].head(10))

    print(f"\nFiles created:")
    print(f"   - patient_static_data.csv")
    print(f"   - synthetic_patient_data.csv")
