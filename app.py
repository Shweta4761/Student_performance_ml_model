"""
Flask REST API for Student Performance Prediction
===================================================
Serves the trained ML models via REST endpoints.
The React frontend calls these endpoints to get predictions + insights.

Endpoints:
  POST /api/predict  — Takes student data, returns predictions + insights
  GET  /api/health   — Health check
  GET  /api/features — Returns expected input features with types/ranges
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from insights import generate_insights, load_reference_data

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)  # Allow React frontend to call directly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# LOAD MODELS (once at startup)
# ============================================================

print("Loading ML models...")
score_model = joblib.load(os.path.join(MODEL_DIR, "score_predictor.pkl"))
grade_model = joblib.load(os.path.join(MODEL_DIR, "grade_predictor.pkl"))
risk_model  = joblib.load(os.path.join(MODEL_DIR, "risk_predictor.pkl"))
dataset_stats, feature_importances = load_reference_data()
print("All models loaded successfully!")

# Feature configuration for the student form
FEATURE_CONFIG = [
    {
        "name": "study_hours_weekly",
        "label": "Study Hours per Week",
        "type": "number",
        "min": 1,
        "max": 40,
        "step": 0.5,
        "placeholder": "e.g., 15",
        "description": "How many hours do you study per week on average?",
        "required": True,
    },
    {
        "name": "attendance_pct",
        "label": "Attendance Percentage",
        "type": "number",
        "min": 0,
        "max": 100,
        "step": 1,
        "placeholder": "e.g., 85",
        "description": "What is your current attendance percentage?",
        "required": True,
    },
    {
        "name": "previous_cgpa",
        "label": "Previous CGPA",
        "type": "number",
        "min": 0,
        "max": 10,
        "step": 0.01,
        "placeholder": "e.g., 7.5 (0 if 1st semester)",
        "description": "Your CGPA from the previous semester (enter 0 if this is your 1st semester).",
        "required": True,
    },
    {
        "name": "sleep_hours",
        "label": "Sleep Hours per Day",
        "type": "number",
        "min": 3,
        "max": 12,
        "step": 0.5,
        "placeholder": "e.g., 7",
        "description": "How many hours do you sleep on an average day?",
        "required": True,
    },
    {
        "name": "mental_health_score",
        "label": "Mental Health Score",
        "type": "number",
        "min": 1,
        "max": 10,
        "step": 1,
        "placeholder": "e.g., 7",
        "description": "Rate your overall mental well-being from 1 (poor) to 10 (excellent).",
        "required": True,
    },
    {
        "name": "has_part_time_job",
        "label": "Part-Time Job",
        "type": "select",
        "options": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
        "description": "Do you currently have a part-time job?",
        "required": True,
    },
    {
        "name": "extracurricular",
        "label": "Extracurricular Activity",
        "type": "select",
        "options": [
            {"value": "No Activity", "label": "No Activity"},
            {"value": "Sports", "label": "Sports"},
            {"value": "Cultural", "label": "Cultural"},
            {"value": "Technical Club", "label": "Technical Club"},
            {"value": "Multiple", "label": "Multiple Activities"},
        ],
        "description": "What type of extracurricular activities are you involved in?",
        "required": True,
    },
    {
        "name": "department",
        "label": "Department",
        "type": "select",
        "options": [
            {"value": "Computer Science", "label": "Computer Science"},
            {"value": "Electronics", "label": "Electronics"},
            {"value": "Mechanical", "label": "Mechanical"},
            {"value": "Civil", "label": "Civil"},
            {"value": "Information Technology", "label": "Information Technology"},
        ],
        "description": "Which department are you in?",
        "required": True,
    },
    {
        "name": "semester",
        "label": "Current Semester",
        "type": "number",
        "min": 1,
        "max": 8,
        "step": 1,
        "placeholder": "e.g., 5",
        "description": "Which semester are you currently in?",
        "required": True,
    },
]


# ============================================================
# ROUTES
# ============================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "message": "Student Performance Prediction API is running."
    })


@app.route("/api/features", methods=["GET"])
def get_features():
    """Return expected input features with their configuration."""
    return jsonify({
        "features": FEATURE_CONFIG,
        "total_features": len(FEATURE_CONFIG),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Accepts student data, returns predictions + insights.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = [f["name"] for f in FEATURE_CONFIG if f.get("required")]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing)}"
            }), 400

        # Prepare input as DataFrame
        student_input = {
            "study_hours_weekly": float(data["study_hours_weekly"]),
            "attendance_pct": float(data["attendance_pct"]),
            "previous_cgpa": float(data["previous_cgpa"]),
            "sleep_hours": float(data["sleep_hours"]),
            "mental_health_score": float(data["mental_health_score"]),
            "has_part_time_job": int(data["has_part_time_job"]),
            "extracurricular": str(data["extracurricular"]),
            "department": str(data["department"]),
            "semester": int(data["semester"]),
        }

        input_df = pd.DataFrame([student_input])

        # Make predictions
        predicted_score = float(score_model.predict(input_df)[0])
        predicted_score = round(max(0, min(100, predicted_score)), 1)

        predicted_grade = str(grade_model.predict(input_df)[0])

        risk_prediction = int(risk_model.predict(input_df)[0])
        risk_probability = float(risk_model.predict_proba(input_df)[0][1])

        # Determine grade confidence (probability of predicted grade)
        grade_probs = grade_model.predict_proba(input_df)[0]
        grade_classes = grade_model.classes_
        grade_confidence = float(max(grade_probs))
        grade_distribution = {
            str(cls): round(float(prob), 3)
            for cls, prob in zip(grade_classes, grade_probs)
        }

        # Generate insights
        insights = generate_insights(
            student_data=student_input,
            predicted_score=predicted_score,
            predicted_grade=predicted_grade,
            at_risk=bool(risk_prediction),
            score_model=score_model,
            stats=dataset_stats,
            importances=feature_importances,
        )

        # Build response
        response = {
            "success": True,
            "predictions": {
                "predicted_score": predicted_score,
                "predicted_grade": predicted_grade,
                "grade_confidence": round(grade_confidence, 3),
                "grade_distribution": grade_distribution,
                "at_risk": bool(risk_prediction),
                "risk_probability": round(risk_probability, 3),
            },
            "insights": insights,
            "input_received": student_input,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Student Performance Prediction API")
    print("  Running on http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
