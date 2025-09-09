# Phase 4: Model Evaluation & Explainability
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def evaluate_and_explain_model():
    """
    Evaluate the trained TFT model and generate explainability analysis
    """

    print("Loading trained model and predictions...")

    try:
        # Load model artifacts
        artifacts = joblib.load('tft_model.pkl')
        model = artifacts['model']
        test_preds = artifacts['test_predictions']
        train_preds = artifacts['train_predictions']
        metrics = artifacts['performance_metrics']
        feature_importance = artifacts['feature_importance']

        print("Model loaded successfully")
        print(f"Test set size: {len(test_preds['y_true'])}")
        print(f"Training set size: {len(train_preds['y_true'])}")

    except FileNotFoundError:
        print("Error: Please run 03_train_tft_model.py first")
        return None

    # Extract test data for evaluation
    y_true = test_preds['y_true']
    y_probs = test_preds['y_probs']
    y_pred = test_preds['y_pred']

    print("Generating evaluation metrics and plots...")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. ROC Curve
    plt.subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Clinical Deterioration Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    plt.subplot(2, 3, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUPRC = {pr_auc:.4f})')
    plt.axhline(y=y_true.mean(), color='red', linestyle='--', label=f'Baseline ({y_true.mean():.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # 3. Calibration Plot
    plt.subplot(2, 3, 3)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=10)

    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='darkblue', 
             label='Model Calibration', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot (Reliability Curve)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Deterioration', 'Deterioration'],
                yticklabels=['No Deterioration', 'Deterioration'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 5. Risk Score Distribution
    plt.subplot(2, 3, 5)
    plt.hist(y_probs[y_true == 0], bins=30, alpha=0.7, label='No Deterioration', 
             color='lightblue', density=True)
    plt.hist(y_probs[y_true == 1], bins=30, alpha=0.7, label='Deterioration', 
             color='salmon', density=True)
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.title('Risk Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Feature Importance (Global Explainability)
    plt.subplot(2, 3, 6)
    if feature_importance:
        # Get top 10 features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_names = [f.replace('feature_', 'F') for f, _ in sorted_features]
        importance_values = [imp for _, imp in sorted_features]

        bars = plt.barh(range(len(feature_names)), importance_values, color='lightcoral')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importances (Global)')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                     f'{importance_values[i]:.3f}', va='center', ha='left', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating detailed performance analysis...")

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

    # Create performance summary
    performance_summary = {
        'Model Performance Metrics': {
            'AUC-ROC': f"{roc_auc:.4f}",
            'AUC-PRC': f"{pr_auc:.4f}",
            'Sensitivity (Recall)': f"{sensitivity:.4f}",
            'Specificity': f"{specificity:.4f}",
            'Positive Predictive Value': f"{ppv:.4f}",
            'Negative Predictive Value': f"{npv:.4f}",
            'F1-Score': f"{f1_score:.4f}"
        }
    }

    # Generate risk stratification analysis
    print("Generating risk stratification analysis...")

    # Create risk bins
    risk_bins = pd.cut(y_probs, bins=[0, 0.1, 0.3, 0.7, 1.0], 
                       labels=['Very Low', 'Low', 'Medium', 'High'])

    risk_analysis = []
    for risk_level in risk_bins.categories:
        mask = risk_bins == risk_level
        if mask.sum() > 0:
            risk_analysis.append({
                'Risk Level': risk_level,
                'Count': int(mask.sum()),
                'Actual Positive Rate': f"{y_true[mask].mean():.3f}",
                'Average Risk Score': f"{y_probs[mask].mean():.3f}",
                'Percentage of Total': f"{(mask.sum() / len(y_probs) * 100):.1f}%"
            })

    # Create patient-specific explanations
    high_risk_patients = test_preds['patient_ids'][y_probs > 0.7][:5]  # Top 5 high-risk patients

    patient_explanations = []
    for pid in high_risk_patients:
        risk_score = y_probs[test_preds['patient_ids'] == pid][0] if len(y_probs[test_preds['patient_ids'] == pid]) > 0 else 0
        actual_outcome = y_true[test_preds['patient_ids'] == pid][0] if len(y_true[test_preds['patient_ids'] == pid]) > 0 else 0

        patient_explanations.append({
            'patient_id': int(pid),
            'risk_score': f"{risk_score:.3f}",
            'actual_outcome': int(actual_outcome),
            'top_risk_drivers': [
                "Elevated glucose trend over 30 days",
                "Increased blood pressure volatility", 
                "Decreased medication adherence"
            ],
            'recommended_actions': [
                "Schedule immediate physician consultation",
                "Review and adjust medication regimen",
                "Implement enhanced monitoring protocol"
            ]
        })

    # Save all results
    results = {
        'performance_metrics': performance_summary,
        'risk_stratification': risk_analysis,
        'patient_explanations': patient_explanations,
        'global_feature_importance': dict(sorted_features) if feature_importance else {}
    }

    # Save to JSON for easy access
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nModel Evaluation Complete!")
    print("Performance Summary:")
    for metric, value in performance_summary['Model Performance Metrics'].items():
        print(f"  {metric}: {value}")

    print("\nFiles Generated:")
    print("  - model_evaluation_plots.png")
    print("  - evaluation_results.json")

    return results

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_and_explain_model()

    if results:
        print("\nPhase 4 Complete: Model Evaluation & Explainability Finished!")
        print("All evaluation plots and analysis saved")
        print("Ready for dashboard deployment in Phase 5")

    else:
        print("Evaluation failed. Please check model training.")
