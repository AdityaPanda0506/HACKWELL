# Phase 2: Data Preprocessing & Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def preprocess_and_engineer_features():
    """
    Load synthetic data and perform comprehensive preprocessing and feature engineering
    """

    print("Loading synthetic data...")

    # Load the datasets
    try:
        static_df = pd.read_csv('patient_static_data.csv')
        ts_df = pd.read_csv('synthetic_patient_data.csv')
        print(f"Loaded {len(static_df)} patients, {len(ts_df)} time series records")
    except FileNotFoundError:
        print("Error: Please run 01_generate_data.py first to create the datasets")
        return None

    print("Merging static and time series data...")

    # Merge static data with time series data
    merged_df = ts_df.merge(static_df, on='patient_id', how='left')

    # Convert 'date' column to datetime
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values(['patient_id', 'day_index'])

    print("Handling missing data...")

    # Handle missing data
    # Forward fill time-series vitals and lab values
    vitals_cols = ['blood_glucose', 'blood_pressure_systolic', 'heart_rate', 'weight', 'bmi',
                   'cholesterol_hdl', 'cholesterol_ldl', 'creatinine', 'hba1c']

    for col in vitals_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df.groupby('patient_id')[col].fillna(method='ffill')

    # Use median imputation for remaining missing values in lifestyle data
    lifestyle_cols = ['daily_steps', 'sleep_hours']
    imputer = SimpleImputer(strategy='median')

    for col in lifestyle_cols:
        if col in merged_df.columns and merged_df[col].isnull().sum() > 0:
            merged_df[col] = imputer.fit_transform(merged_df[[col]]).flatten()

    print("Creating engineered features...")

    # Feature Engineering
    # Sort by patient and day for proper window calculations
    merged_df = merged_df.sort_values(['patient_id', 'day_index'])

    # Initialize feature columns list
    feature_cols = []

    # 1. Rolling window features for key vitals
    vitals_for_features = ['blood_glucose', 'blood_pressure_systolic', 'heart_rate']

    def calculate_trend(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        try:
            slope = np.polyfit(x, series, 1)[0]
            return slope
        except:
            return 0

    for vital in vitals_for_features:
        if vital in merged_df.columns:
            # 7-day rolling mean
            merged_df[f'{vital}_7d_mean'] = merged_df.groupby('patient_id')[vital].rolling(window=7, min_periods=1).mean().values

            # 14-day rolling std (volatility)
            merged_df[f'{vital}_14d_std'] = merged_df.groupby('patient_id')[vital].rolling(window=14, min_periods=2).std().fillna(0).values

            # 30-day trend (slope of linear regression)
            merged_df[f'{vital}_30d_trend'] = merged_df.groupby('patient_id')[vital].rolling(window=30, min_periods=5).apply(calculate_trend).values

            # Percentage change from 30 days ago
            merged_df[f'{vital}_30d_pct_change'] = merged_df.groupby('patient_id')[vital].pct_change(periods=30).fillna(0)

            # Append feature columns
            feature_cols.extend([f'{vital}_7d_mean', f'{vital}_14d_std', f'{vital}_30d_trend', f'{vital}_30d_pct_change'])

    # 2. Medication adherence features
    if 'medication_adherence' in merged_df.columns:
        # 7-day average adherence
        merged_df['med_adherence_7d_avg'] = merged_df.groupby('patient_id')['medication_adherence'].rolling(window=7, min_periods=1).mean().values

        # Days since poor adherence (< 0.7)
        merged_df['days_since_poor_adherence'] = 0
        for pid in merged_df['patient_id'].unique():
            patient_data = merged_df[merged_df['patient_id'] == pid].copy()
            poor_adherence_days = patient_data[patient_data['medication_adherence'] < 0.7].index

            for idx in patient_data.index:
                if len(poor_adherence_days) > 0:
                    last_poor_day = poor_adherence_days[poor_adherence_days <= idx]
                    if len(last_poor_day) > 0:
                        days_since = idx - last_poor_day.max()
                        merged_df.loc[idx, 'days_since_poor_adherence'] = days_since
                    else:
                        merged_df.loc[idx, 'days_since_poor_adherence'] = 999  # Never had poor adherence
                else:
                    merged_df.loc[idx, 'days_since_poor_adherence'] = 999

        feature_cols.extend(['med_adherence_7d_avg', 'days_since_poor_adherence'])

    # Fill any remaining NaN values
    merged_df = merged_df.fillna(0)

    # Create final feature set including base features
    base_features = ['patient_id', 'day_index', 'age', 'blood_glucose', 'blood_pressure_systolic', 
                     'heart_rate', 'weight', 'bmi', 'medication_adherence', 'daily_steps', 
                     'sleep_hours', 'cholesterol_hdl', 'cholesterol_ldl', 'creatinine', 'hba1c']

    existing_base_features = [f for f in base_features if f in merged_df.columns]
    existing_feature_cols = [f for f in feature_cols if f in merged_df.columns]

    final_feature_set = existing_base_features + existing_feature_cols + ['deterioration_in_next_90d', 'deterioration_event']

    # Create final processed dataset
    processed_df = merged_df[final_feature_set].copy()

    # Save processed data to CSV
    processed_df.to_csv('processed_patient_data.csv', index=False)

    print(f"\nPreprocessing Complete!")
    print(f"Total patients: {processed_df['patient_id'].nunique()}")
    print(f"Total records: {len(processed_df)}")
    print(f"Features created: {len(existing_feature_cols)}")
    print(f"File saved: processed_patient_data.csv")

    return processed_df

if __name__ == "__main__":
    processed_df = preprocess_and_engineer_features()

    if processed_df is not None:
        print(f"\nSample of processed data:")
        print(processed_df[['patient_id', 'day_index', 'blood_glucose', 'deterioration_in_next_90d']].head(10))
