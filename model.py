# model.py
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class SimplifiedTFTModel:
    """
    Simplified TFT-inspired model using GradientBoostingClassifier
    """

    def __init__(self, max_encoder_length=30, random_state=42):
        self.max_encoder_length = max_encoder_length
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state
        )
        self.feature_names = None
        self.feature_importance_dict = {}

    def prepare_sequences(self, df, target_col='deterioration_in_next_90d'):
        print(f"Preparing sequences with encoder length: {self.max_encoder_length}")

        sequences = []
        targets = []
        patient_ids = []

        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id].sort_values('day_index')

            for i in range(self.max_encoder_length, len(patient_data)):
                sequence = patient_data.iloc[i-self.max_encoder_length:i]
                target_row = patient_data.iloc[i]

                sequence_features = self._extract_sequence_features(sequence)

                sequences.append(sequence_features)
                targets.append(target_row[target_col])
                patient_ids.append(patient_id)

        return np.array(sequences), np.array(targets), np.array(patient_ids)

    def _extract_sequence_features(self, sequence):
        features = []

        numeric_cols = [
            'blood_glucose', 'blood_pressure_systolic', 'heart_rate', 'weight', 'bmi',
            'medication_adherence', 'daily_steps', 'sleep_hours',
            'cholesterol_hdl', 'cholesterol_ldl', 'creatinine', 'hba1c'
        ]

        engineered_cols = [col for col in sequence.columns 
                          if any(suffix in col for suffix in ['_7d_mean', '_14d_std', '_30d_trend', '_pct_change'])]

        all_feature_cols = [col for col in numeric_cols + engineered_cols if col in sequence.columns]

        for col in all_feature_cols:
            values = sequence[col].fillna(sequence[col].median())
            features.extend([
                values.mean(),
                values.std() if len(values) > 1 else 0,
                values.iloc[-1],
                values.iloc[-1] - values.iloc[0] if len(values) > 1 else 0,
                values.max(),
                values.min(),
            ])

        if 'age' in sequence.columns:
            features.append(sequence['age'].iloc[-1])

        return features

    def fit(self, X, y):
        print(f"Training model on {X.shape[0]} sequences...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_dict = {
                f'feature_{i}': importance 
                for i, importance in enumerate(self.model.feature_importances_)
            }
        print("Model training completed!")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
