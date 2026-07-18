import json
import re

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from groq import Groq

from models import Hospital, MedicalHistory, db

prediction_bp = Blueprint("prediction", __name__)

all_symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic_patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]

SYSTEM_PROMPT = """You are an expert medical triage assistant.
Your job is to analyze symptoms and return a probable disease, a confidence score, a short explanation, matched symptoms, a doctor type, and clear do's and don'ts.

Rules:
- Return valid JSON only
- Keep the explanation short and easy to understand
- Use the symptoms provided by the user only
- If the symptom set is unclear, set confidence low and explain that more symptoms are needed
- Include a concise recommendation list under recommendations.do, recommendations.dont, and recommendations.try

Return exactly this JSON:
{
  "disease": "Disease name",
  "confidence": 0,
  "verdict": "most probably / likely / possibly / need more symptoms",
  "doctor": "Specialist type",
  "explanation": "Short reasoning",
  "matched_symptoms": ["symptom1", "symptom2"],
  "recommendations": {
    "do": ["..."],
    "dont": ["..."],
    "try": ["..."]
  },
  "secondary_possibilities": [
    {
      "disease": "Alternative disease",
      "why": "Short reason"
    }
  ]
}"""


def _display_name(symptom: str) -> str:
    return symptom.replace("_", " ").title()


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _groq_client() -> Groq:
    return Groq(api_key=current_app.config["GROQ_API_KEY"])


def _prediction_label(top_prediction):
    disease = top_prediction.get("disease", "")
    verdict = str(top_prediction.get("verdict", "")).strip().lower()
    confidence = float(top_prediction.get("confidence", 0) or 0)
    state = top_prediction.get("state", "")

    if disease in {"Need more symptoms for a reliable match", "Unknown / Insufficient Symptoms", "Unknown"} or state == "need_more_symptoms":
        return "Need more symptoms"
    if verdict == "most probably":
        return "Most probably"
    if verdict == "likely":
        return "Likely"
    if verdict == "possibly":
        return "Possibly"
    if verdict == "need more symptoms":
        return "Need more symptoms"
    if confidence >= 92:
        return "Most probably"
    if confidence >= 84:
        return "Likely"
    if confidence > 0:
        return "Possibly"
    return "Review"


def _recommendations_from_prediction(prediction):
    recommendations = prediction.get("recommendations") or {}
    return {
        "do": recommendations.get(
            "do",
            [
                "Keep monitoring your symptoms and rest well.",
                "Stay hydrated and seek care if symptoms worsen.",
            ],
        ),
        "dont": recommendations.get(
            "dont",
            [
                "Do not ignore severe symptoms or sudden worsening.",
                "Do not self-medicate without medical advice.",
            ],
        ),
        "try": recommendations.get(
            "try",
            [
                "Book a doctor visit if symptoms continue.",
                "Retest with a clearer symptom set if needed.",
            ],
        ),
    }


def _normalize_prediction(prediction, selected_symptoms):
    confidence = float(prediction.get("confidence", 0) or 0)
    verdict = str(prediction.get("verdict", "")).strip().lower()
    disease = str(prediction.get("disease", "Unknown")).strip() or "Unknown"
    doctor = str(prediction.get("doctor", "General physician")).strip() or "General physician"
    explanation = str(prediction.get("explanation", "")).strip()
    matched_symptoms = prediction.get("matched_symptoms", selected_symptoms)
    if not isinstance(matched_symptoms, list):
        matched_symptoms = selected_symptoms
    matched_symptoms = [item for item in matched_symptoms if isinstance(item, str) and item in all_symptoms]
    if not matched_symptoms:
        matched_symptoms = selected_symptoms

    state = prediction.get("state")
    if not state:
        if disease.lower().startswith("need more symptoms") or verdict == "need more symptoms":
            state = "need_more_symptoms"
        elif verdict in {"most probably", "likely"}:
            state = "high_confidence"
        elif verdict == "possibly":
            state = "mixed_symptoms"
        elif confidence >= 80:
            state = "high_confidence"
        else:
            state = "mixed_symptoms"

    return {
        "disease": disease,
        "confidence": confidence,
        "doctor": doctor,
        "explanation": explanation or "More symptoms are needed for a reliable analysis.",
        "matched_symptoms": matched_symptoms,
        "recommendations": _recommendations_from_prediction(prediction),
        "secondary_possibilities": prediction.get("secondary_possibilities", []),
        "state": state,
        "verdict": verdict or _prediction_label({"disease": disease, "confidence": confidence, "verdict": verdict, "state": state}),
    }


@prediction_bp.route("/symptoms", methods=["GET", "POST"])
@login_required
def select_symptoms():
    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")
        if not selected_symptoms:
            flash("Please select at least one symptom.")
            return redirect(url_for("prediction.select_symptoms"))

        client = _groq_client()
        symptoms_readable = ", ".join(_display_name(symptom) for symptom in selected_symptoms)
        prompt = (
            "The user selected these symptoms: "
            f"{symptoms_readable}. "
            "Predict the most likely disease and return JSON matching the schema exactly."
        )

        try:
            completion = client.chat.completions.create(
                model=current_app.config["GROQ_MODEL"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            raw = completion.choices[0].message.content or ""
            prediction = _parse_json_response(raw)
        except Exception as exc:
            flash(f"Groq API error: {exc}")
            return redirect(url_for("prediction.select_symptoms"))

        normalized = _normalize_prediction(prediction, selected_symptoms)
        top_prediction = normalized
        hospital = None
        if top_prediction["state"] == "high_confidence":
            hospital = Hospital.query.filter(
                Hospital.specialist_type.ilike(f"%{top_prediction['doctor'].split(' / ')[0]}%")
            ).first()
        hospital_name = hospital.name if hospital else "Please consult a nearby general hospital."

        new_history = MedicalHistory(
            user_id=current_user.id,
            predicted_disease=top_prediction["disease"],
            confidence=top_prediction["confidence"],
            prediction_state=top_prediction["state"],
            recommended_doctor=top_prediction["doctor"],
            recommended_hospital=hospital_name,
        )
        new_history.set_symptoms(selected_symptoms)
        db.session.add(new_history)
        db.session.commit()

        return render_template(
            "prediction_result.html",
            predictions=[top_prediction],
            doctor=top_prediction["doctor"],
            hospital=hospital_name,
            symptoms=selected_symptoms,
            is_reliable_match=top_prediction["state"] == "high_confidence",
            result_state=top_prediction["state"],
            confidence_label=_prediction_label(top_prediction),
            explanation=top_prediction.get("explanation"),
            matched_symptoms=top_prediction.get("matched_symptoms", []),
            recommendations=top_prediction.get("recommendations", {}),
            secondary_possibilities=top_prediction.get("secondary_possibilities", []),
            confidence=top_prediction.get("confidence", 0),
        )

    return render_template("symptom_selection.html", symptoms=all_symptoms)


@prediction_bp.route("/history")
@login_required
def history():
    user_history = MedicalHistory.query.filter_by(user_id=current_user.id).order_by(MedicalHistory.timestamp.desc()).all()
    return render_template("history.html", history=user_history)


@prediction_bp.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    MedicalHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("Medical history cleared successfully.")
    return redirect(url_for("prediction.history"))
