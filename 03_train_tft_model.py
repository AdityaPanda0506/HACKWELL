# 03_train_tft_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
import warnings
from model import SimplifiedTFTModel

warnings.filterwarnings('ignore')

def train_tft_model():
    print("Loading processed data...")
    try:
        df = pd.read_csv('processed_patient_data.csv')
        print(f"Loaded {len(df)} records for {df['patient_id'].nunique()} patients")
    except FileNotFoundError:
        print("Error: Please run 02_preprocess_data.py first")
        return None

    tft_model = SimplifiedTFTModel(max_encoder_length=30)
    X, y, patient_ids = tft_model.prepare_sequences(df)

    print(f"Created {len(X)} training sequences")
    print(f"Feature vector size: {X.shape[1] if len(X.shape) > 1 else 0}")
    print(f"Positive class ratio: {y.mean():.1%}")

    if len(X) == 0:
        print("No sequences created. Check data integrity.")
        return None

    unique_patients = np.unique(patient_ids)
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Training set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")

    tft_model.fit(X_train, y_train)

    print("Making predictions...")
    train_probs = tft_model.predict_proba(X_train)
    test_probs = tft_model.predict_proba(X_test)

    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    train_auprc = average_precision_score(y_train, train_probs)
    test_auprc = average_precision_score(y_test, test_probs)

    print("\nModel Performance:")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Training AUPRC: {train_auprc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_preds))

    model_artifacts = {
        'model': tft_model,
        'test_predictions': {
            'patient_ids': patient_ids[test_mask],
            'y_true': y_test,
            'y_probs': test_probs,
            'y_pred': test_preds
        },
        'train_predictions': {
            'patient_ids': patient_ids[train_mask],
            'y_true': y_train,
            'y_probs': train_probs,
            'y_pred': train_preds
        },
        'performance_metrics': {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_auprc': train_auprc,
            'test_auprc': test_auprc
        },
        'feature_importance': tft_model.feature_importance_dict
    }

    joblib.dump(model_artifacts, 'tft_model.pkl')

    predictions_df = pd.DataFrame({
        'patient_id': np.concatenate([patient_ids[train_mask], patient_ids[test_mask]]),
        'split': ['train'] * len(y_train) + ['test'] * len(y_test),
        'y_true': np.concatenate([y_train, y_test]),
        'risk_score': np.concatenate([train_probs, test_probs]),
        'prediction': np.concatenate([train_preds, test_preds])
    })

    predictions_df['risk_level'] = pd.cut(predictions_df['risk_score'], bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])

    predictions_df.to_csv('model_predictions.csv', index=False)

    print("\nModel saved to: tft_model.pkl")
    print("Predictions saved to: model_predictions.csv")
    print("Phase 3 Complete: TFT Model Training Finished!")

    return tft_model, model_artifacts

if __name__ == "__main__":
    model, artifacts = train_tft_model()
    if model is not None:
        print("\nTop Feature Importances:")
        feature_importance = artifacts['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"  {feature}: {importance:.4f}")
        print("\nModel ready for evaluation and dashboard deployment!")
    else:
        print("Model training failed. Please check data and try again.")
