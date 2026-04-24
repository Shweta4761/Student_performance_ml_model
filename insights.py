"""
Insight Generation Module
==========================
Generates actionable insights from the trained ML models:
  - Weak area detection (comparison to top performers)
  - Study time recommendations (simulated improvement)
  - Risk factor explanations
  - Degree/field suggestions based on strengths
"""

import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_reference_data():
    """Load dataset stats and feature importances."""
    with open(os.path.join(MODEL_DIR, "dataset_stats.json"), "r") as f:
        stats = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "r") as f:
        importances = json.load(f)
    return stats, importances


def detect_weak_areas(student_data, stats):
    """
    Compare student's inputs against top-performer averages.
    Returns a list of areas where the student is significantly below.
    """
    weak_areas = []

    comparisons = {
        "study_hours_weekly": {
            "label": "Study Hours",
            "unit": "hrs/week",
            "higher_is_better": True,
            "threshold_pct": 0.25,  # flag if 25% below top performers
        },
        "attendance_pct": {
            "label": "Attendance",
            "unit": "%",
            "higher_is_better": True,
            "threshold_pct": 0.10,
        },
        "previous_cgpa": {
            "label": "Previous CGPA",
            "unit": "",
            "higher_is_better": True,
            "threshold_pct": 0.15,
        },
        "sleep_hours": {
            "label": "Sleep Hours",
            "unit": "hrs/day",
            "higher_is_better": True,
            "threshold_pct": 0.15,
        },
        "mental_health_score": {
            "label": "Mental Health",
            "unit": "/10",
            "higher_is_better": True,
            "threshold_pct": 0.15,
        },
    }

    for feature, config in comparisons.items():
        if feature not in student_data or feature not in stats:
            continue

        student_val = float(student_data[feature])
        top_avg = stats[feature]["top_performer_mean"]
        overall_avg = stats[feature]["mean"]

        if config["higher_is_better"]:
            gap = top_avg - student_val
            pct_below = gap / top_avg if top_avg > 0 else 0

            if pct_below > config["threshold_pct"]:
                weak_areas.append({
                    "area": config["label"],
                    "current": round(student_val, 1),
                    "recommended": round(top_avg, 1),
                    "overall_avg": round(overall_avg, 1),
                    "message": f"Your {config['label'].lower()} is {student_val:.1f}{config['unit']}. "
                               f"Top performers average {top_avg:.1f}{config['unit']}. "
                               f"Try to increase to at least {round(top_avg * 0.9, 1)}{config['unit']}."
                })

    return weak_areas


def generate_study_recommendation(student_data, score_model, stats):
    """
    Simulate increasing study hours and predict the score improvement.
    """
    current_hours = float(student_data.get("study_hours_weekly", 10))
    top_hours = stats.get("study_hours_weekly", {}).get("top_performer_mean", 22)

    # Simulate predictions at different study hours
    simulations = []
    for hours in [current_hours, current_hours + 5, current_hours + 10, top_hours]:
        sim_data = student_data.copy()
        sim_data["study_hours_weekly"] = min(hours, 40)
        sim_df = pd.DataFrame([sim_data])
        predicted_score = score_model.predict(sim_df)[0]
        simulations.append({
            "hours": round(min(hours, 40), 1),
            "predicted_score": round(predicted_score, 1)
        })

    current_score = simulations[0]["predicted_score"]
    best_score = simulations[-1]["predicted_score"]
    improvement = best_score - current_score

    if improvement > 10:
        message = (f"Increasing study time from {current_hours:.0f} to {top_hours:.0f} hrs/week "
                   f"could improve your score by ~{improvement:.0f} points (from {current_score:.0f} to {best_score:.0f}).")
    elif improvement > 5:
        message = (f"Adding {top_hours - current_hours:.0f} more study hours per week "
                   f"could give you a ~{improvement:.0f} point boost.")
    else:
        message = (f"Your study hours are good! Focus on quality over quantity. "
                   f"Consider improving attendance or sleep for better results.")

    return {
        "current_hours": round(current_hours, 1),
        "recommended_hoursapp": round(top_hours, 1),
        "simulations": simulations,
        "potential_improvement": round(improvement, 1),
        "message": message,
    }


def identify_strengths(student_data, stats):
    """Identify areas where the student is performing well."""
    strengths = []

    checks = {
        "attendance_pct": ("Consistent Attendance", 80),
        "study_hours_weekly": ("Strong Study Habits", 18),
        "sleep_hours": ("Healthy Sleep Pattern", 7),
        "mental_health_score": ("Good Mental Health Balance", 7),
        "previous_cgpa": ("Strong Academic Foundation", 7.5),
    }

    for feature, (label, threshold) in checks.items():
        if feature in student_data and float(student_data[feature]) >= threshold:
            strengths.append(label)

    if int(student_data.get("has_part_time_job", 0)) == 0:
        strengths.append("Focused on Academics (no part-time job)")

    extra = student_data.get("extracurricular", "No Activity")
    if extra != "No Activity":
        strengths.append(f"Active in Extracurriculars ({extra})")

    return strengths


def identify_risk_factors(student_data, at_risk, stats):
    """Explain why a student might be at risk."""
    factors = []

    if not at_risk:
        return factors

    if float(student_data.get("attendance_pct", 100)) < 60:
        factors.append("Low attendance (below 60%) is a strong risk indicator.")

    if float(student_data.get("study_hours_weekly", 20)) < 10:
        factors.append("Study hours are significantly below average.")

    if float(student_data.get("previous_cgpa", 10)) < 5.0 and int(student_data.get("semester", 1)) > 1:
        factors.append("Previous CGPA indicates past academic struggles.")

    if float(student_data.get("mental_health_score", 10)) < 5:
        factors.append("Low mental health score — consider reaching out to counseling services.")

    if float(student_data.get("sleep_hours", 8)) < 5:
        factors.append("Insufficient sleep can significantly impact academic performance.")

    if not factors:
        factors.append("Multiple factors combined indicate moderate risk. Focus on improving study hours and attendance.")

    return factors


def suggest_fields(student_data, predicted_score, predicted_grade):
    """
    Suggest future degree/career fields based on department,
    predicted performance, and student profile.
    """
    department = student_data.get("department", "")
    cgpa = float(student_data.get("previous_cgpa", 0))
    extra = student_data.get("extracurricular", "No Activity")
    study_hours = float(student_data.get("study_hours_weekly", 10))

    suggestions = []

    # Department-specific suggestions
    dept_fields = {
        "Computer Science": [
            {"field": "Software Development / Full-Stack", "base_match": "High",
             "reason": "Core CS skills are directly applicable to software engineering careers."},
            {"field": "Data Science / AI / ML", "base_match": "High",
             "reason": "CS background is ideal for data science and machine learning roles."},
            {"field": "M.Tech / MS in CS", "base_match": "Medium",
             "reason": "Strong academic foundation supports graduate studies."},
            {"field": "Cybersecurity", "base_match": "Medium",
             "reason": "Growing field with high demand for CS graduates."},
        ],
        "Electronics": [
            {"field": "VLSI / Embedded Systems", "base_match": "High",
             "reason": "Direct application of electronics knowledge."},
            {"field": "IoT / Robotics", "base_match": "High",
             "reason": "Electronics background is essential for hardware-software integration."},
            {"field": "M.Tech in ECE", "base_match": "Medium",
             "reason": "Higher studies to specialize in advanced electronics."},
            {"field": "Telecom / Signal Processing", "base_match": "Medium",
             "reason": "Core electronics concepts align well with telecom."},
        ],
        "Mechanical": [
            {"field": "Design & Manufacturing", "base_match": "High",
             "reason": "Core mechanical engineering career path."},
            {"field": "Automotive Industry", "base_match": "High",
             "reason": "Growing sector with demand for mechanical engineers."},
            {"field": "M.Tech / ME", "base_match": "Medium",
             "reason": "Specialize in thermal, design, or production engineering."},
            {"field": "Renewable Energy", "base_match": "Medium",
             "reason": "Emerging field combining mechanical principles with sustainability."},
        ],
        "Civil": [
            {"field": "Structural Engineering", "base_match": "High",
             "reason": "Core civil engineering specialization."},
            {"field": "Construction Management", "base_match": "High",
             "reason": "Combines technical knowledge with project management."},
            {"field": "M.Tech in Civil", "base_match": "Medium",
             "reason": "Advanced studies in structural or environmental engineering."},
            {"field": "Urban Planning", "base_match": "Medium",
             "reason": "Apply civil engineering knowledge to city development."},
        ],
        "Information Technology": [
            {"field": "Web / App Development", "base_match": "High",
             "reason": "IT background aligns perfectly with development roles."},
            {"field": "Cloud Computing / DevOps", "base_match": "High",
             "reason": "High-demand field for IT graduates."},
            {"field": "M.Tech / MCA", "base_match": "Medium",
             "reason": "Graduate studies to deepen technical expertise."},
            {"field": "Business Analysis / IT Consulting", "base_match": "Medium",
             "reason": "Combine technical knowledge with business acumen."},
        ],
    }

    # Get department suggestions or generic ones
    base_suggestions = dept_fields.get(department, [
        {"field": "Higher Studies (M.Tech / MS)", "base_match": "Medium",
         "reason": "Graduate studies for specialization."},
        {"field": "Industry Roles", "base_match": "Medium",
         "reason": "Apply your skills in the industry."},
    ])

    for sugg in base_suggestions:
        match = sugg["base_match"]

        # Upgrade/downgrade based on predicted performance
        if predicted_grade in ["A+", "A"] and match == "Medium":
            match = "High"
        elif predicted_grade in ["D", "F"] and match == "High":
            match = "Medium"

        # Higher studies boost if high CGPA
        if "M.Tech" in sugg["field"] or "MS" in sugg["field"]:
            if cgpa >= 8.0 and predicted_grade in ["A+", "A", "B+"]:
                match = "High"
                sugg["reason"] += " Your strong CGPA makes you a competitive candidate."
            elif cgpa < 6.0:
                match = "Low"

        # Technical club boosts technical fields
        if extra == "Technical Club" and any(kw in sugg["field"].lower() for kw in ["software", "data", "iot", "vlsi", "development"]):
            if match == "Medium":
                match = "High"
            sugg["reason"] += " Active in technical clubs shows practical interest."

        suggestions.append({
            "field": sugg["field"],
            "match": match,
            "reason": sugg["reason"],
        })

    # Sort by match level
    match_order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions.sort(key=lambda x: match_order.get(x["match"], 3))

    return suggestions[:4]  # Return top 4


def generate_insights(student_data, predicted_score, predicted_grade, at_risk,
                      score_model, stats, importances):
    """
    Master function: Generate all insights for a student.
    Returns a comprehensive insights dictionary.
    """
    weak_areas = detect_weak_areas(student_data, stats)
    study_rec = generate_study_recommendation(student_data, score_model, stats)
    strengths = identify_strengths(student_data, stats)
    risk_factors = identify_risk_factors(student_data, at_risk, stats)
    field_suggestions = suggest_fields(student_data, predicted_score, predicted_grade)

    return {
        "weak_areas": weak_areas,
        "study_recommendation": study_rec,
        "strengths": strengths,
        "risk_factors": risk_factors,
        "field_suggestions": field_suggestions,
    }
