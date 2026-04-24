"""
Student Performance Model Training Pipeline
=============================================
Trains 3 ML models on the student_performance.csv dataset:
  1. Random Forest Regressor  → predicts final_score
  2. Random Forest Classifier → predicts grade
  3. Random Forest Classifier → predicts at_risk (binary)

Uses only the features a student would provide (no exam scores),
so the model works for "early prediction" before exams.

Saves all artifacts (models, preprocessor, feature importances)
to the `models/` directory for use by the Flask API.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Features the student will provide (7 key inputs + department & semester for context)
STUDENT_INPUT_FEATURES = [
    "study_hours_weekly",    # How many hours do you study per week?
    "attendance_pct",        # What is your attendance %?
    "previous_cgpa",         # Your previous CGPA (0 if semester 1)
    "sleep_hours",           # How many hours do you sleep daily?
    "mental_health_score",   # Rate your mental health (1-10)
    "has_part_time_job",     # Do you have a part-time job? (0/1)
    "extracurricular",       # Extracurricular activity type
    "department",            # Your department
    "semester",              # Current semester
]

NUMERICAL_FEATURES = [
    "study_hours_weekly", "attendance_pct", "previous_cgpa",
    "sleep_hours", "mental_health_score", "semester"
]

CATEGORICAL_FEATURES = [
    "has_part_time_job", "extracurricular", "department"
]

# Target columns
TARGET_SCORE = "final_score"
TARGET_GRADE = "grade"
TARGET_RISK = "at_risk"

GRADE_ORDER = ["F", "D", "C", "B", "B+", "A", "A+"]


def load_data():
    """Load and prepare the dataset."""
    print("=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Using features: {STUDENT_INPUT_FEATURES}")
    return df


def build_preprocessor():
    """Build a ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def train_score_model(X_train, X_test, y_train, y_test, preprocessor):
    """Train Random Forest Regressor for final_score prediction."""
    print("\n" + "=" * 60)
    print("  MODEL 1: Final Score Predictor (Regression)")
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

    print(f"  R² Score:       {r2:.4f}")
    print(f"  MAE:            {mae:.2f}")
    print(f"  RMSE:           {rmse:.2f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = {
        "model": "RandomForestRegressor",
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
    }

    return pipeline, metrics


def train_grade_model(X_train, X_test, y_train, y_test, preprocessor):
    """Train Random Forest Classifier for grade prediction."""
    print("\n" + "=" * 60)
    print("  MODEL 2: Grade Predictor (Multiclass Classification)")
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  F1 (weighted):      {f1:.4f}")
    print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "model": "RandomForestClassifier",
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
    }

    return pipeline, metrics


def train_risk_model(X_train, X_test, y_train, y_test, preprocessor):
    """Train Random Forest Classifier for at_risk binary prediction."""
    print("\n" + "=" * 60)
    print("  MODEL 3: At-Risk Predictor (Binary Classification)")
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  CV F1 (5-fold):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "model": "RandomForestClassifier (balanced)",
        "accuracy": round(accuracy, 4),
        "f1_binary": round(f1, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }

    return pipeline, metrics


def extract_feature_importances(score_pipeline, preprocessor_fitted):
    """Extract and save feature importances from the score model."""
    regressor = score_pipeline.named_steps["regressor"]
    importances = regressor.feature_importances_

    # Get feature names after preprocessing
    feature_names = preprocessor_fitted.get_feature_names_out()
    # Clean up names (remove prefixes like "num__" and "cat__")
    feature_names = [name.split("__")[-1] for name in feature_names]

    importance_dict = dict(zip(feature_names, importances.tolist()))
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[-1], reverse=True))

    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCES (Score Model)")
    print("=" * 60)
    for feat, imp in importance_dict.items():
        bar = "#" * int(imp * 100)
        print(f"  {feat:35s} {imp:.4f} {bar}")

    return importance_dict


def compute_dataset_stats(df):
    """Compute reference statistics for insight generation."""
    stats = {}

    # Top performer thresholds (students with grade A or A+)
    top_students = df[df["grade"].isin(["A", "A+"])]

    for col in NUMERICAL_FEATURES:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "top_performer_mean": round(float(top_students[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
        }

    return stats


def main():
    # Load data
    df = load_data()

    # Prepare features and targets
    X = df[STUDENT_INPUT_FEATURES]
    y_score = df[TARGET_SCORE]
    y_grade = df[TARGET_GRADE]
    y_risk = df[TARGET_RISK]

    # Split data (same split for all 3 models for consistency)
    X_train, X_test, y_score_train, y_score_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )
    y_grade_train = y_grade.loc[X_train.index]
    y_grade_test = y_grade.loc[X_test.index]
    y_risk_train = y_risk.loc[X_train.index]
    y_risk_test = y_risk.loc[X_test.index]

    # Build preprocessor
    preprocessor = build_preprocessor()

    # Train all 3 models
    score_model, score_metrics = train_score_model(
        X_train, X_test, y_score_train, y_score_test, preprocessor
    )
    grade_model, grade_metrics = train_grade_model(
        X_train, X_test, y_grade_train, y_grade_test, build_preprocessor()
    )
    risk_model, risk_metrics = train_risk_model(
        X_train, X_test, y_risk_train, y_risk_test, build_preprocessor()
    )

    # Extract feature importances
    fitted_preprocessor = score_model.named_steps["preprocessor"]
    importances = extract_feature_importances(score_model, fitted_preprocessor)

    # Compute dataset statistics for insight generation
    dataset_stats = compute_dataset_stats(df)

    # ============================================================
    # SAVE ALL ARTIFACTS
    # ============================================================
    print("\n" + "=" * 60)
    print("  SAVING MODEL ARTIFACTS")
    print("=" * 60)

    # Save models
    joblib.dump(score_model, os.path.join(MODEL_DIR, "score_predictor.pkl"), compress=3)
    print("  [OK] Saved score_predictor.pkl")

    joblib.dump(grade_model, os.path.join(MODEL_DIR, "grade_predictor.pkl"), compress=3)
    print("  [OK] Saved grade_predictor.pkl")

    joblib.dump(risk_model, os.path.join(MODEL_DIR, "risk_predictor.pkl"), compress=3)
    print("  [OK] Saved risk_predictor.pkl")

    # Save feature importances
    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)
    print("  [OK] Saved feature_importance.json")

    # Save dataset stats
    with open(os.path.join(MODEL_DIR, "dataset_stats.json"), "w") as f:
        json.dump(dataset_stats, f, indent=2)
    print("  [OK] Saved dataset_stats.json")

    # Save training report
    training_report = {
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features_used": STUDENT_INPUT_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "score_model": score_metrics,
        "grade_model": grade_metrics,
        "risk_model": risk_metrics,
    }
    with open(os.path.join(MODEL_DIR, "training_report.json"), "w") as f:
        json.dump(training_report, f, indent=2)
    print("  [OK] Saved training_report.json")

    print("\n" + "=" * 60)
    print("  ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n  Artifacts saved to: {MODEL_DIR}")
    print()


if __name__ == "__main__":
    main()
