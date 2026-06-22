import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

def run_training_pipeline():
    print("--- Starting Phase 2 & 3: Training & Calibration ---")
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_dir, 'Dataset', 'clean_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("clean_data.csv not found. Please run data_prep.py first.")

    # 1. Load Data
    df = pd.read_csv(data_path)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    X = df.drop(columns=['target', 'encounter_id'])
    y = df['target']
    groups = df['patient_nbr']

    # 2. Train/Test Split (Prevent Leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    
    print(f"Training on {len(X_train)} encounters. Testing on {len(X_test)}.")

    # 3. Feature Selection ("Fairness Through Unawareness")
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if 'race' in categorical_cols: categorical_cols.remove('race')
    if 'gender' in categorical_cols: categorical_cols.remove('gender')

    # 4. Build Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop" # Safely ignores race and gender
    )

    scale_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )

    uncalib_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])

    # 5. Train Base Model
    print("Training base XGBoost model...")
    uncalib_pipeline.fit(X_train, y_train)
    joblib.dump(uncalib_pipeline, os.path.join(models_dir, 'xgb_pipeline.pkl'))

    # 6. Calibrate Model (Phase 3)
    print("Calibrating probabilities using Sigmoid Scaling...")
    calib_model = CalibratedClassifierCV(estimator=uncalib_pipeline, method='sigmoid', cv=5, n_jobs=-1)
    calib_model.fit(X_train, y_train)
    joblib.dump(calib_model, os.path.join(models_dir, 'calibrated_xgb_pipeline.pkl'))
    
    # 7. Evaluate
    y_prob = calib_model.predict_proba(X_test)[:, 1]
    overall_auc = roc_auc_score(y_test, y_prob)
    print(f"✅ Final Test AUC: {overall_auc:.4f}")
    
    # 8. Fairness Audit (Phase 5 check)
    print("\n--- Running Fairness Audit ---")
    X_test['predicted_risk'] = y_prob
    X_test['actual_readmission'] = y_test
    X_test['race_gender'] = X_test['race'].astype(str) + " _ " + X_test['gender'].astype(str)
    
    failed = False
    for group in X_test['race_gender'].unique():
        sub = X_test[X_test['race_gender'] == group]
        if len(sub) > 50 and len(sub['actual_readmission'].unique()) > 1:
            sub_auc = roc_auc_score(sub['actual_readmission'], sub['predicted_risk'])
            if sub_auc - overall_auc < -0.05:
                print(f"❌ FAIRNESS FAILED: {group} AUC gap is {sub_auc - overall_auc:.4f}")
                failed = True
                
    if not failed:
        print("✅ FAIRNESS AUDIT PASSED: No subgroup fell below -0.05 threshold.")
    else:
        print("WARNING: Fairness audit failed. Review model deployment constraints.")

if __name__ == "__main__":
    run_training_pipeline()
