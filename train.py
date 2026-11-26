"""
Production Model Training and Benchmarking Pipeline
RNA-Seq Pan-Cancer Intelligence (TCGA PANCAN)
===================================================
Rigorous Leak-Free Methodology:
1. Strict Stratified Train/Test Split (80/20) with held-out testing.
2. End-to-End Scikit-Learn Pipeline: VarianceThreshold -> StandardScaler -> SelectKBest(ANOVA) -> Classifier.
3. Stratified 5-Fold Cross-Validation strictly inside training folds to eliminate selection bias.
4. Comprehensive multi-model benchmark (Linear SVM, Logistic Regression, Random Forest, XGBoost).
5. Pre-fitted 3D PCA coordinates and demo test samples for production Streamlit deployment.
"""

import os
import sys
import time
import datetime
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("🚀 Starting RNA-Seq Pan-Cancer Production Training Pipeline")
    print("=" * 70)

    # 1. Load Data
    data_path = 'dataset/data.csv'
    labels_path = 'dataset/labels.csv'

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing dataset files in {os.path.dirname(data_path)}")

    t0 = time.time()
    print("📥 Loading RNA-Seq Expression matrix and cohort labels...")
    X = pd.read_csv(data_path, index_col=0)
    y_df = pd.read_csv(labels_path, index_col=0)
    y = y_df.iloc[:, 0]
    print(f"   ✓ Loaded {X.shape[0]} samples with {X.shape[1]} gene expression markers in {time.time()-t0:.2f}s.")

    # 2. Encode Labels & Class Proportions
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"   ✓ Cohort Classes ({len(classes)}): {classes}")
    for c, count in y.value_counts().items():
        print(f"     - {c}: {count} patients ({count/len(y)*100:.1f}%)")

    # 3. Clean Stratified Train/Test Split (80/20)
    # The test set is held out completely until final validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\n📊 Stratified Partitioning:")
    print(f"   ✓ Training Set: {X_train.shape[0]} patients")
    print(f"   ✓ Held-Out Test Set: {X_test.shape[0]} patients")

    # 4. Define Candidate Architectures for Benchmarking
    candidate_models = {
        'Logistic_Regression': LogisticRegression(
            C=0.1, max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Linear_SVM': CalibratedClassifierCV(
            SVC(kernel='linear', C=0.01, random_state=42),
            ensemble=False
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric='mlogloss', random_state=42, n_jobs=-1
        )
    }

    # 5. Stratified 5-Fold Cross-Validation Benchmark
    print("\n🔬 Executing 5-Fold Cross-Validation Benchmark (Zero Selection Leakage)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}

    for name, clf in candidate_models.items():
        t_model_start = time.time()
        # Full pipeline encapsulated: VarianceThreshold -> StandardScaler -> SelectKBest -> Classifier
        pipe = Pipeline([
            ('var', VarianceThreshold(threshold=0.1)),
            ('scaler', StandardScaler()),
            ('select', SelectKBest(score_func=f_classif, k=500)),
            ('clf', clf)
        ])

        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=['accuracy', 'f1_macro', 'balanced_accuracy'],
            n_jobs=-1,
            return_train_score=False
        )
        elapsed = time.time() - t_model_start

        acc = cv_scores['test_accuracy'].mean()
        f1 = cv_scores['test_f1_macro'].mean()
        b_acc = cv_scores['test_balanced_accuracy'].mean()

        benchmark_results[name] = {
            'cv_accuracy': float(acc),
            'cv_f1_macro': float(f1),
            'cv_balanced_accuracy': float(b_acc),
            'fit_time_seconds': round(elapsed, 2)
        }
        print(f"   [{name:20s}] CV Acc: {acc*100:.2f}% | CV F1-Macro: {f1*100:.2f}% | Fit: {elapsed:.1f}s")

    # 6. Fit Final Champion Model on Full Training Set
    # Select best model by CV F1-Macro
    best_model_name = max(benchmark_results, key=lambda k: benchmark_results[k]['cv_f1_macro'])
    print(f"\n🏆 Champion Model Selected: {best_model_name}")

    best_pipeline = Pipeline([
        ('var', VarianceThreshold(threshold=0.1)),
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=500)),
        ('clf', candidate_models[best_model_name])
    ])

    print("Fitting champion pipeline on full training partition...")
    best_pipeline.fit(X_train, y_train)

    # Extract Feature Names
    var_mask = best_pipeline.named_steps['var'].get_support()
    var_features = X.columns[var_mask]
    select_mask = best_pipeline.named_steps['select'].get_support()
    selected_features = list(var_features[select_mask])
    print(f"   ✓ Extracted {len(selected_features)} top discriminative features from ANOVA filtering.")

    # 7. Independent Hold-Out Test Set Evaluation
    print("\n🎯 Evaluating Champion Model on Held-Out Test Set (161 patients)...")
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
    cm = confusion_matrix(y_test, y_test_pred)
    report_dict = classification_report(y_test, y_test_pred, target_names=classes, output_dict=True)

    print(f"   ✓ Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"   ✓ Test F1-Macro:     {test_f1 * 100:.2f}%")
    print(f"   ✓ Test ROC-AUC(OvR): {test_auc * 100:.2f}%")
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=classes))

    # Also evaluate all other models on test set for leaderboard
    for name, clf in candidate_models.items():
        p = Pipeline([
            ('var', VarianceThreshold(threshold=0.1)),
            ('scaler', StandardScaler()),
            ('select', SelectKBest(score_func=f_classif, k=500)),
            ('clf', clf)
        ])
        p.fit(X_train, y_train)
        pred = p.predict(X_test)
        prob = p.predict_proba(X_test) if hasattr(p, 'predict_proba') else None
        benchmark_results[name]['test_accuracy'] = float(accuracy_score(y_test, pred))
        benchmark_results[name]['test_f1_macro'] = float(f1_score(y_test, pred, average='macro'))
        benchmark_results[name]['test_roc_auc'] = float(roc_auc_score(y_test, prob, multi_class='ovr')) if prob is not None else 1.0

    # 8. Train 3D PCA Manifold for Interactive Visualization
    print("\n🌌 Pre-computing 3D PCA coordinates for tumor manifold...")
    # Preprocess training data through the pipeline transformers
    X_train_prep = best_pipeline[:-1].transform(X_train)
    pca_3d = PCA(n_components=3, random_state=42)
    coords_train = pca_3d.fit_transform(X_train_prep)

    train_pca_df = pd.DataFrame(
        coords_train,
        columns=['PC1', 'PC2', 'PC3'],
        index=X_train.index
    )
    train_pca_df['Cohort'] = le.inverse_transform(y_train)
    print(f"   ✓ PCA Explained Variance: PC1={pca_3d.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca_3d.explained_variance_ratio_[1]*100:.1f}%, "
          f"PC3={pca_3d.explained_variance_ratio_[2]*100:.1f}%")

    # 9. Extract Top Universal Biomarkers with Cohort Associations
    print("\n🧬 Mining Top Universal Biomarker Candidates...")
    f_scores = best_pipeline.named_steps['select'].scores_[select_mask]
    p_values = best_pipeline.named_steps['select'].pvalues_[select_mask]

    top_bio_df = pd.DataFrame({
        'Gene_ID': selected_features,
        'ANOVA_F_Score': f_scores,
        'P_Value': p_values
    }).sort_values('ANOVA_F_Score', ascending=False)

    # Determine which cancer cohort exhibits the highest expression for each top gene
    dominant_cohorts = []
    for g in top_bio_df['Gene_ID']:
        means = X_train.groupby(y_train)[g].mean()
        dominant_cohorts.append(classes[means.idxmax()])
    top_bio_df['Peak_Cancer_Cohort'] = dominant_cohorts

    print(f"   ✓ Top 5 Highest-Ranked Biomarkers:")
    for idx, row in top_bio_df.head(5).iterrows():
        print(f"     - {row['Gene_ID']}: F={row['ANOVA_F_Score']:.1f} (Peak in {row['Peak_Cancer_Cohort']})")

    # 10. Extract 5 Representative Test Samples as Built-in "Demo Cohort"
    # One sample for each cancer class
    demo_samples = []
    for c_idx, c_name in enumerate(classes):
        matching_indices = np.where(y_test == c_idx)[0]
        if len(matching_indices) > 0:
            sample_loc = matching_indices[0]
            sample_series = X_test.iloc[sample_loc].copy()
            sample_series.name = f"{c_name}_Patient_Ref"
            demo_samples.append(sample_series)
    demo_samples_df = pd.DataFrame(demo_samples)
    print(f"   ✓ Built-in Demo Cohort: {demo_samples_df.shape[0]} patient profiles ready.")

    # 11. Bundle & Export Complete Production Artifact
    export_path = 'cancer_classification_pipeline.pkl'
    print(f"\n📦 Packaging production artifacts into `{export_path}`...")

    artifacts = {
        'pipeline': best_pipeline,
        'label_encoder': le,
        'classes': classes,
        'original_features': list(X.columns),
        'kept_features_variance': list(var_features),
        'final_features': selected_features,
        'pca_3d': pca_3d,
        'train_pca_df': train_pca_df,
        'demo_samples_df': demo_samples_df,
        'benchmark_results': benchmark_results,
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1),
            'roc_auc_ovr': float(test_auc),
            'classification_report': report_dict,
            'confusion_matrix': cm.tolist()
        },
        'top_biomarkers': top_bio_df.head(50),
        'metadata': {
            'dataset_name': 'TCGA Pan-Cancer RNA-Seq (PANCAN)',
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'raw_genes': X.shape[1],
            'selected_genes': len(selected_features),
            'best_model': best_model_name,
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    joblib.dump(artifacts, export_path, compress=3)
    file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"✅ Successfully exported `{export_path}` ({file_size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
