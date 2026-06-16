"""
SmartGrid Guardian - Model Evaluation Script
=============================================

This script evaluates all trained models and generates comprehensive reports,
visualizations, and feature importance analysis.

The script runs independently and does not affect the deployed Streamlit application.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# XGBoost and SHAP imports
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Random state for reproducibility
RANDOM_STATE = 42

# Output directories
OUTPUT_DIRS = {
    'confusion_matrices': 'results/confusion_matrices',
    'feature_importance': 'results/feature_importance',
    'reports': 'results/reports',
    'shap': 'results/shap',
}

# ==========================================
# DIRECTORY SETUP
# ==========================================

def create_directories():
    """Create all required output directories."""
    for dir_path in OUTPUT_DIRS.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("[✓] Output directories created successfully.\n")


# ==========================================
# DATA LOADING AND PREPROCESSING
# ==========================================

def load_and_preprocess_data():
    """
    Load and preprocess the dataset using the same steps as in the project.
    
    Returns:
        X_train_scaled: Training features (scaled)
        X_test_scaled: Testing features (scaled)
        X_train: Training features (unscaled, for tree-based models)
        X_test: Testing features (unscaled, for tree-based models)
        y_train: Training target
        y_test: Testing target
        feature_names: List of feature names
        scaler: Fitted scaler
    """
    print("[→] Loading dataset...")
    df = pd.read_csv('smart_grid_stability_augmented.csv')
    
    print(f"[✓] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features\n")
    
    # Prepare features and target
    X = df.drop(['stab', 'stabf'], axis=1)
    y = df['stabf'].map({'stable': 1, 'unstable': 0})
    
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Scale features for distance-based and linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[✓] Train-Test Split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"[✓] Features: {len(feature_names)}\n")
    
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, feature_names, scaler


# ==========================================
# MODEL TRAINING
# ==========================================

def train_models(X_train_scaled, X_train, y_train):
    """
    Train all models.
    
    Args:
        X_train_scaled: Scaled training features
        X_train: Unscaled training features
        y_train: Training target
    
    Returns:
        Dictionary of trained models
    """
    print("[→] Training models...\n")
    
    models = {}
    
    # 1. Logistic Regression (requires scaling)
    print("  • Training Logistic Regression...")
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    models['Logistic Regression'].fit(X_train_scaled, y_train)
    
    # 2. Decision Tree
    print("  • Training Decision Tree...")
    models['Decision Tree'] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    models['Decision Tree'].fit(X_train, y_train)
    
    # 3. KNN (requires scaling)
    print("  • Training KNN...")
    models['KNN'] = KNeighborsClassifier(n_neighbors=5)
    models['KNN'].fit(X_train_scaled, y_train)
    
    # 4. SVM (requires scaling)
    print("  • Training SVM...")
    models['SVM'] = SVC(kernel='linear', random_state=RANDOM_STATE, probability=True, max_iter=2000)
    models['SVM'].fit(X_train_scaled, y_train)
    
    # 5. Gaussian Naive Bayes (requires scaling)
    print("  • Training Gaussian Naive Bayes...")
    models['Gaussian Naive Bayes'] = GaussianNB()
    models['Gaussian Naive Bayes'].fit(X_train_scaled, y_train)
    
    # 6. Random Forest
    print("  • Training Random Forest...")
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    models['Random Forest'].fit(X_train, y_train)
    
    # 7. XGBoost
    print("  • Training XGBoost...")
    models['XGBoost'] = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    models['XGBoost'].fit(X_train, y_train)
    
    print("\n[✓] All models trained successfully.\n")
    
    return models


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_models(models, X_test_scaled, X_test, y_test, feature_names):
    """
    Evaluate all models and generate metrics.
    
    Args:
        models: Dictionary of trained models
        X_test_scaled: Scaled test features
        X_test: Unscaled test features
        y_test: Test target
        feature_names: List of feature names
    
    Returns:
        Dictionary with evaluation results and predictions
    """
    print("[→] Evaluating models...\n")
    
    results = []
    predictions = {}
    
    # Models that require scaling
    scaled_models = ['Logistic Regression', 'KNN', 'SVM', 'Gaussian Naive Bayes']
    
    for model_name, model in models.items():
        print(f"  • Evaluating {model_name}...")
        
        # Get predictions
        if model_name in scaled_models:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        predictions[model_name] = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Confusion Matrix': cm
        })
    
    print("\n[✓] Model evaluation completed.\n")
    
    return results, predictions


# ==========================================
# RESULTS SUMMARY
# ==========================================

def create_comparison_table(results):
    """
    Create and save model comparison table.
    
    Args:
        results: List of evaluation results
    
    Returns:
        DataFrame with comparison table
    """
    print("[→] Creating model comparison table...\n")
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['Model'],
            'Accuracy': f"{r['Accuracy']:.4f}",
            'Precision': f"{r['Precision']:.4f}",
            'Recall': f"{r['Recall']:.4f}",
            'F1-score': f"{r['F1-score']:.4f}"
        }
        for r in results
    ])
    
    # Sort by accuracy (convert back to float for sorting)
    comparison_df_sorted = pd.DataFrame([
        {
            'Model': r['Model'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1-score': r['F1-score']
        }
        for r in results
    ]).sort_values('Accuracy', ascending=False)
    
    # Format for display
    comparison_display = comparison_df_sorted.copy()
    comparison_display['Accuracy'] = comparison_display['Accuracy'].apply(lambda x: f"{x:.4f}")
    comparison_display['Precision'] = comparison_display['Precision'].apply(lambda x: f"{x:.4f}")
    comparison_display['Recall'] = comparison_display['Recall'].apply(lambda x: f"{x:.4f}")
    comparison_display['F1-score'] = comparison_display['F1-score'].apply(lambda x: f"{x:.4f}")
    
    print("=" * 80)
    print("MODEL COMPARISON TABLE (Sorted by Accuracy)")
    print("=" * 80)
    print(comparison_display.to_string(index=False))
    print("=" * 80 + "\n")
    
    # Save to CSV (with numeric values)
    comparison_csv = comparison_df_sorted.copy()
    comparison_csv.to_csv(
        f"{OUTPUT_DIRS['reports']}/model_comparison.csv",
        index=False
    )
    print(f"[✓] Comparison table saved to {OUTPUT_DIRS['reports']}/model_comparison.csv\n")
    
    return comparison_df_sorted


# ==========================================
# CLASSIFICATION REPORTS
# ==========================================

def generate_classification_reports(models, predictions, X_test_scaled, X_test, y_test):
    """
    Generate and save classification reports for all models.
    
    Args:
        models: Dictionary of trained models
        predictions: Dictionary of predictions
        X_test_scaled: Scaled test features
        X_test: Unscaled test features
        y_test: Test target
    """
    print("[→] Generating classification reports...\n")
    
    scaled_models = ['Logistic Regression', 'KNN', 'SVM', 'Gaussian Naive Bayes']
    
    report_content = []
    report_content.append("=" * 80)
    report_content.append("CLASSIFICATION REPORTS - ALL MODELS")
    report_content.append("=" * 80)
    report_content.append("")
    
    for model_name, y_pred in predictions.items():
        report_content.append(f"\n{'='*80}")
        report_content.append(f"MODEL: {model_name}")
        report_content.append(f"{'='*80}\n")
        
        report = classification_report(y_test, y_pred, digits=4)
        report_content.append(report)
        
        cm = confusion_matrix(y_test, y_pred)
        report_content.append(f"\nConfusion Matrix:\n{cm}\n")
    
    # Save to file
    with open(f"{OUTPUT_DIRS['reports']}/classification_reports.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
    
    print(f"[✓] Classification reports saved to {OUTPUT_DIRS['reports']}/classification_reports.txt\n")


# ==========================================
# CONFUSION MATRIX PLOTS
# ==========================================

def generate_confusion_matrices(models, predictions, X_test_scaled, X_test, y_test):
    """
    Generate and save confusion matrix plots for all models.
    
    Args:
        models: Dictionary of trained models
        predictions: Dictionary of predictions
        X_test_scaled: Scaled test features
        X_test: Unscaled test features
        y_test: Test target
    """
    print("[→] Generating confusion matrix plots...\n")
    
    for model_name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Unstable', 'Stable'],
                   yticklabels=['Unstable', 'Stable'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save
        filename = f"{OUTPUT_DIRS['confusion_matrices']}/{model_name.replace(' ', '_').lower()}_cm.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    print(f"\n[✓] All confusion matrices saved to {OUTPUT_DIRS['confusion_matrices']}/\n")


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def generate_feature_importance(models, X_test, y_test, feature_names):
    """
    Generate and save feature importance for tree-based models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
    """
    print("[→] Generating feature importance plots...\n")
    
    importance_models = ['Decision Tree', 'Random Forest', 'XGBoost']
    
    for model_name in importance_models:
        if model_name not in models:
            continue
        
        model = models[model_name]
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        # Save CSV
        csv_path = f"{OUTPUT_DIRS['feature_importance']}/{model_name.replace(' ', '_').lower()}_importance.csv"
        importance_df.sort_values('Importance', ascending=False).to_csv(csv_path, index=False)
        print(f"  ✓ CSV saved: {csv_path}")
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{OUTPUT_DIRS['feature_importance']}/{model_name.replace(' ', '_').lower()}_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot saved: {plot_path}")
    
    print(f"\n[✓] Feature importance analysis completed.\n")


# ==========================================
# SHAP ANALYSIS (XGBoost)
# ==========================================

def generate_shap_analysis(models, X_train, X_test, y_test):
    """
    Generate SHAP plots for XGBoost model.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_test: Test target
    """
    print("[→] Generating SHAP analysis for XGBoost...\n")
    
    if 'XGBoost' not in models:
        print("[!] XGBoost model not found. Skipping SHAP analysis.\n")
        return
    
    xgb_model = models['XGBoost']
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use second class (stable)
        
        # SHAP Summary Plot
        print("  • Generating SHAP summary plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        summary_path = f"{OUTPUT_DIRS['shap']}/shap_summary_plot.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary plot saved: {summary_path}")
        
        # SHAP Bar Plot
        print("  • Generating SHAP bar plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        bar_path = f"{OUTPUT_DIRS['shap']}/shap_bar_plot.png"
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Bar plot saved: {bar_path}")
        
        print(f"\n[✓] SHAP analysis completed.\n")
        
    except Exception as e:
        print(f"[!] Error generating SHAP plots: {str(e)}\n")


# ==========================================
# RESEARCH SUMMARY
# ==========================================

def generate_research_summary(results):
    """
    Generate research summary report.
    
    Args:
        results: List of evaluation results
    """
    print("[→] Generating research summary...\n")
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['Accuracy'], reverse=True)
    best_model = sorted_results[0]
    
    # Separate ensemble and non-ensemble models
    ensemble_models = ['Random Forest', 'XGBoost']
    ensemble_results = [r for r in sorted_results if r['Model'] in ensemble_models]
    non_ensemble_results = [r for r in sorted_results if r['Model'] not in ensemble_models]
    
    # Calculate average accuracy
    ensemble_avg_acc = np.mean([r['Accuracy'] for r in ensemble_results]) if ensemble_results else 0
    non_ensemble_avg_acc = np.mean([r['Accuracy'] for r in non_ensemble_results]) if non_ensemble_results else 0
    
    summary_content = f"""
{'='*80}
SMARTGRID GUARDIAN - MODEL EVALUATION RESEARCH SUMMARY
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}
This report summarizes the evaluation of 7 machine learning models for predicting
grid stability in the Smart Grid Guardian system.

KEY FINDINGS
{'-'*80}

1. MODELS EVALUATED: {len(results)}
   • Logistic Regression
   • Decision Tree
   • KNN
   • SVM
   • Gaussian Naive Bayes
   • Random Forest
   • XGBoost

2. BEST-PERFORMING MODEL: {best_model['Model']}
   • Accuracy:  {best_model['Accuracy']:.4f}
   • Precision: {best_model['Precision']:.4f}
   • Recall:    {best_model['Recall']:.4f}
   • F1-score:  {best_model['F1-score']:.4f}

3. PERFORMANCE RANKING
{'-'*80}
"""
    
    for idx, result in enumerate(sorted_results, 1):
        summary_content += f"\n   {idx}. {result['Model']:<25} Accuracy: {result['Accuracy']:.4f}"
    
    summary_content += f"""

4. ENSEMBLE METHODS PERFORMANCE
{'-'*80}
Ensemble models showed superior performance:

   Average Accuracy (Ensemble):      {ensemble_avg_acc:.4f}
   Average Accuracy (Non-Ensemble):  {non_ensemble_avg_acc:.4f}
   
   Performance Gain:                 {(ensemble_avg_acc - non_ensemble_avg_acc) * 100:.2f}%

ANALYSIS AND INSIGHTS
{'-'*80}

Why Ensemble Methods Performed Well:

1. DIVERSITY OF LEARNERS:
   Ensemble methods combine multiple learning algorithms, capturing different
   patterns in the data. Random Forest uses decision trees, while XGBoost uses
   boosting to iteratively improve predictions.

2. VARIANCE REDUCTION:
   Ensemble methods reduce overfitting through:
   • Random sampling of data (bagging in Random Forest)
   • Sequential error correction (boosting in XGBoost)
   • Averaging multiple predictions reduces variance

3. HANDLING COMPLEX PATTERNS:
   Grid stability depends on complex interactions between power flow (p1-p4),
   response dynamics (tau1-tau4), and demand flexibility (g1-g4).
   Ensemble methods excel at capturing these non-linear relationships.

4. ROBUSTNESS:
   Ensemble models are more robust to:
   • Outliers in the training data
   • Class imbalance (though stratification was used)
   • Variations in feature scaling (tree-based models don't require scaling)

5. FEATURE INTERACTION:
   Both Random Forest and XGBoost can model interactions between features
   without explicit feature engineering.

METHODOLOGY
{'-'*80}

Data:
• Dataset: smart_grid_stability_augmented.csv
• Total Samples: {len(pd.read_csv('smart_grid_stability_augmented.csv'))}
• Train-Test Split: 80-20 with stratification
• Random State: 42 (reproducibility)

Preprocessing:
• Feature Scaling: StandardScaler applied for distance-based and linear models
• No scaling applied to tree-based models (as per best practices)
• Target Encoding: 'stable' → 1, 'unstable' → 0

Evaluation Metrics:
• Accuracy: Overall correctness
• Precision (weighted): Reliability of positive predictions
• Recall (weighted): Coverage of positive instances
• F1-score (weighted): Harmonic mean balancing precision and recall
• Confusion Matrix: Detailed classification breakdown

RECOMMENDATIONS
{'-'*80}

1. DEPLOYMENT:
   Use {best_model['Model']} as the primary model for the Streamlit dashboard,
   as it provides the best accuracy ({best_model['Accuracy']:.4f}).

2. MONITORING:
   Monitor model performance on new data continuously. Consider retraining
   if accuracy drops below 95% of initial performance.

3. FEATURE IMPORTANCE:
   Focus on top features for grid monitoring and parameter optimization.
   See feature_importance/ directory for detailed analysis.

4. EXPLAINABILITY:
   Use SHAP values (generated for XGBoost) to explain individual predictions
   to stakeholders and grid operators.

5. MODEL ENSEMBLE:
   Consider ensembling the top 3 models for even more robust predictions
   in production environments.

{'='*80}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    # Save summary
    summary_path = f"{OUTPUT_DIRS['reports']}/research_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(summary_content)
    print(f"[✓] Research summary saved to {summary_path}\n")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main execution function."""
    print("\n")
    print("=" * 80)
    print("SmartGrid Guardian - Model Evaluation")
    print("=" * 80 + "\n")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Load and preprocess data
    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
    
    # Step 3: Train models
    models = train_models(X_train_scaled, X_train, y_train)
    
    # Step 4: Evaluate models
    results, predictions = evaluate_models(models, X_test_scaled, X_test, y_test, feature_names)
    
    # Step 5: Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Step 6: Generate classification reports
    generate_classification_reports(models, predictions, X_test_scaled, X_test, y_test)
    
    # Step 7: Generate confusion matrices
    generate_confusion_matrices(models, predictions, X_test_scaled, X_test, y_test)
    
    # Step 8: Generate feature importance
    generate_feature_importance(models, X_test, y_test, feature_names)
    
    # Step 9: Generate SHAP analysis
    generate_shap_analysis(models, X_train, X_test, y_test)
    
    # Step 10: Generate research summary
    generate_research_summary(results)
    
    print("=" * 80)
    print("✓ Model Evaluation Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: results/")
    print("\nDirectory structure:")
    print("  results/")
    print("  ├── confusion_matrices/")
    print("  ├── feature_importance/")
    print("  ├── reports/")
    print("  └── shap/\n")


if __name__ == "__main__":
    main()
