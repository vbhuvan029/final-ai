import json
import os
import re
import urllib.error
import urllib.request

import numpy as np


BASE_DIR = os.path.dirname(__file__)
SYMPTOMS_PATH = os.path.join(BASE_DIR, "symptoms.json")
DOCTOR_MAPPING_PATH = os.path.join(BASE_DIR, "doctor_mapping.json")
PROFILES_PATH = os.path.join(BASE_DIR, "disease_profiles.json")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile"
GROQ_MODEL_CANDIDATES = [
    name
    for name in dict.fromkeys(
        [
            GROQ_MODEL,
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "llama3-70b-8192",
        ]
    )
    if name
]


try:
    with open(SYMPTOMS_PATH, "r") as f:
        all_symptoms = json.load(f)
    with open(DOCTOR_MAPPING_PATH, "r") as f:
        doctor_mapping = json.load(f)
    with open(PROFILES_PATH, "r") as f:
        disease_profiles = json.load(f)
except FileNotFoundError:
    print("Warning: symptom profile files not found.")
    all_symptoms, doctor_mapping, disease_profiles = [], {}, {}


PROFILE_WEIGHT = 0.90
INVALID_DIAGNOSES = {"Unknown", "Unknown / Insufficient Symptoms", "Need more symptoms for a reliable match"}
MIN_SYMPTOMS_FOR_LABEL = 3

DISEASE_RECOMMENDATIONS = {
    "Heart attack": {
        "do": [
            "Seek emergency care immediately if chest pain is severe or spreading.",
            "Rest and avoid exertion until a doctor evaluates you.",
            "Take only medicines prescribed for you by a clinician.",
        ],
        "dont": [
            "Do not ignore chest pain, sweating, or breathlessness.",
            "Do not drive yourself if symptoms are severe.",
            "Do not wait for symptoms to fully settle before seeking help.",
        ],
        "try": [
            "If the pain is ongoing, call emergency services right away.",
            "Keep someone with you while waiting for care.",
        ],
    },
    "Diabetes": {
        "do": [
            "Monitor blood sugar regularly and follow your medication plan.",
            "Eat balanced meals with controlled portions.",
            "Stay hydrated and keep a log of symptoms.",
        ],
        "dont": [
            "Do not skip meals and then overeat later.",
            "Do not ignore blurry vision, extreme thirst, or frequent urination.",
            "Do not change insulin or medicines without medical advice.",
        ],
        "try": [
            "Take a light walk after meals if your doctor allows it.",
            "Track glucose readings and share trends with your doctor.",
        ],
    },
    "AIDS": {
        "do": [
            "See a specialist for proper evaluation and confirmed testing.",
            "Take prescribed medicines exactly as directed.",
            "Maintain good nutrition and adequate rest.",
        ],
        "dont": [
            "Do not self-diagnose from symptoms alone.",
            "Do not share needles or personal items.",
            "Do not delay testing if you have high-risk symptoms.",
        ],
        "try": [
            "Book a medical consultation for confirmatory testing.",
            "Keep a record of fever, weight loss, and recurrent infections.",
        ],
    },
    "Pneumonia": {
        "do": [
            "Rest and drink enough fluids unless a doctor told you otherwise.",
            "Use prescribed antibiotics or inhalers exactly as directed.",
            "Watch for worsening breathlessness or fever.",
        ],
        "dont": [
            "Do not smoke or expose yourself to smoke.",
            "Do not ignore shortness of breath or chest pain.",
            "Do not stop medication early.",
        ],
        "try": [
            "Use a humid environment if it helps breathing.",
            "Follow up if fever or cough does not improve quickly.",
        ],
    },
    "Allergy": {
        "do": [
            "Avoid the trigger if you know what caused it.",
            "Use allergy medicines only as prescribed or recommended.",
            "Rest and monitor for swelling or breathing trouble.",
        ],
        "dont": [
            "Do not re-expose yourself to the suspected trigger.",
            "Do not ignore wheezing, lip swelling, or breathing difficulty.",
            "Do not mix medicines without checking labels.",
        ],
        "try": [
            "Wash exposed skin and change clothes after a trigger exposure.",
            "If symptoms worsen, seek medical care promptly.",
        ],
    },
    "Drug Reaction": {
        "do": [
            "Stop the suspected medicine only if a doctor advises you to.",
            "Contact a clinician quickly for review of the reaction.",
            "Keep a list of all medicines you took recently.",
        ],
        "dont": [
            "Do not take the same suspected drug again until cleared.",
            "Do not ignore rash with fever or facial swelling.",
            "Do not self-treat severe reactions at home.",
        ],
        "try": [
            "Photograph the rash to show your doctor.",
            "Carry the medicine name when you visit the clinic.",
        ],
    },
    "Bronchial Asthma": {
        "do": [
            "Use your inhaler exactly as prescribed.",
            "Avoid dust, smoke, and known triggers.",
            "Sit upright and rest if you feel short of breath.",
        ],
        "dont": [
            "Do not overexert yourself during an attack.",
            "Do not ignore wheezing or frequent night symptoms.",
            "Do not skip controller medicines if prescribed.",
        ],
        "try": [
            "Use a rescue inhaler if you have one and it was prescribed.",
            "Seek urgent care if breathing gets worse quickly.",
        ],
    },
    "General": {
        "do": [
            "Keep a symptom diary and monitor changes.",
            "Rest, hydrate, and eat lightly until you are reviewed.",
            "See a doctor if symptoms persist or worsen.",
        ],
        "dont": [
            "Do not ignore persistent pain, fever, or breathing issues.",
            "Do not self-medicate with multiple medicines at once.",
            "Do not wait too long if symptoms are getting worse.",
        ],
        "try": [
            "Book a clinic visit for a proper evaluation.",
            "Bring this symptom list to your appointment.",
        ],
    },
}


def _build_input_vector(user_symptoms):
    input_vector = np.zeros(len(all_symptoms))
    valid_symptoms = []
    for symptom in user_symptoms:
        clean = symptom.strip()
        if clean in all_symptoms:
            input_vector[all_symptoms.index(clean)] = 1
            valid_symptoms.append(clean)
    return input_vector, valid_symptoms


def _score_profiles(valid_symptoms):
    profile_scores = {}
    for disease, profile in disease_profiles.items():
        probable = [all_symptoms[i] for i, p in enumerate(profile["symptom_probs"]) if p >= 0.25]
        core = profile["core_symptoms"]
        if not probable:
            continue

        broad_matches = sum(1 for symptom in valid_symptoms if symptom in probable)
        core_matches = sum(1 for symptom in valid_symptoms if symptom in core)
        if broad_matches == 0:
            continue

        coverage = broad_matches / len(valid_symptoms)
        core_coverage = core_matches / max(len(core), 1)
        specificity = broad_matches / len(probable)
        core_density = core_matches / len(valid_symptoms)
        profile_scores[disease] = (
            (coverage * 0.35)
            + (core_coverage * 0.35)
            + (specificity * 0.10)
            + (core_density * 0.20)
        )
    return profile_scores


def _combine_scores(valid_symptoms, input_vector):
    profile_scores = _score_profiles(valid_symptoms)
    combined = {}
    total_weights = {}

    for disease in disease_profiles.keys():
        combined[disease] = profile_scores.get(disease, 0.0) * PROFILE_WEIGHT
        total_weights[disease] = PROFILE_WEIGHT

    for disease in list(combined.keys()):
        weight_total = total_weights.get(disease, 1.0)
        combined[disease] = combined[disease] / max(weight_total, 1e-9)

    ranked = sorted(
        ((disease, score) for disease, score in combined.items() if disease not in INVALID_DIAGNOSES),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked, profile_scores


def _strip_code_fences(content):
    content = str(content or "").strip()
    if content.startswith("```"):
        parts = content.split("```", 2)
        if len(parts) >= 3:
            content = parts[1] + "\n" + parts[2]
        content = content.strip()
    if content.startswith("json"):
        content = content[4:].strip()
    return content


def _parse_json_content(content):
    content = _strip_code_fences(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _as_text_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        pieces = [piece.strip(" -•\t") for piece in re.split(r"[\n;|]+", value)]
        return [piece for piece in pieces if piece]
    return []


def _recommend_need_more_symptoms():
    return {
        "disease": "Need more symptoms for a reliable match",
        "confidence": 0.0,
        "doctor": "General Physician",
        "state": "need_more_symptoms",
        "verdict": "Need more symptoms",
        "matched_symptoms": [],
        "explanation": {
            "summary": "More symptoms are needed before naming a disease.",
            "why_it_matches": "The symptom set is too broad or not specific enough for a reliable match.",
            "what_to_watch": "Add symptoms that feel more specific to the current issue.",
            "when_to_seek_help": "See a clinician if symptoms are severe, persistent, or getting worse.",
        },
        "recommendations": get_disease_recommendations("General"),
    }


def _has_enough_support(disease, valid_symptoms):
    profile = disease_profiles.get(disease, {})
    core = profile.get("core_symptoms", [])
    probable = [all_symptoms[i] for i, p in enumerate(profile.get("symptom_probs", [])) if p >= 0.25]
    core_matches = sum(1 for symptom in valid_symptoms if symptom in core)
    probable_matches = sum(1 for symptom in valid_symptoms if symptom in probable)

    if len(valid_symptoms) < MIN_SYMPTOMS_FOR_LABEL:
        return False
    if core_matches == 0 and probable_matches < 2:
        return False
    if disease == "Allergy" and core_matches == 0:
        return False
    return True


def _groq_request(messages, max_tokens=520, temperature=0.25, timeout=40):
    if not GROQ_API_KEY:
        return None

    last_error = None
    for model_name in GROQ_MODEL_CANDIDATES:
        for use_json_mode in (True, False):
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}

            request = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                if content:
                    return _parse_json_content(content)
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                continue

    if last_error:
        raise last_error
    return None


def _groq_refine_prediction(valid_symptoms, ranked_candidates, local_confidence, best_disease, best_score, best_gap, core_matches, coverage_ratio):
    if not GROQ_API_KEY or not ranked_candidates:
        return None

    candidates = ranked_candidates[:5]
    structured_candidates = []
    for disease, score in candidates:
        if disease in INVALID_DIAGNOSES:
            continue
        profile = disease_profiles.get(disease, {})
        probable = [all_symptoms[i] for i, p in enumerate(profile.get("symptom_probs", [])) if p >= 0.25]
        core = profile.get("core_symptoms", [])
        structured_candidates.append({
            "disease": disease,
            "local_score": round(float(score), 4),
            "doctor": doctor_mapping.get(disease, "General Physician"),
            "core_symptoms": core[:10],
            "probable_symptoms": probable[:16],
            "matched_core_symptoms": [symptom for symptom in valid_symptoms if symptom in core],
            "matched_probable_symptoms": [symptom for symptom in valid_symptoms if symptom in probable],
        })

    if not structured_candidates:
        return None

    prompt_data = {
        "selected_symptoms": valid_symptoms,
        "local_signal": {
            "best_disease": best_disease,
            "core_matches": int(core_matches),
            "coverage_ratio": round(float(coverage_ratio), 4),
        },
        "candidate_diseases": structured_candidates,
        "response_contract": {
            "verdict": "most probably | likely | possibly | need more symptoms",
            "state": "high_confidence | mixed_symptoms | need_more_symptoms",
            "disease": "exact disease label or Need more symptoms for a reliable match",
            "doctor": "specialist name",
            "matched_symptoms": "array of symptoms copied from selected_symptoms",
            "explanation": {
                "summary": "short direct summary",
                "why_it_matches": "short paragraph",
                "what_to_watch": "short warning sentence",
                "when_to_seek_help": "short escalation sentence",
            },
            "recommendations": {
                "do": ["2-4 immediate actions"],
                "dont": ["2-4 avoid actions"],
                "try": ["1-3 practical next steps"],
            },
        },
        "rules": [
            "Use only the selected symptoms and candidate diseases.",
            "Choose need_more_symptoms if the evidence is too mixed, too broad, or too few symptoms are present.",
            "Do not return numbers or percentages.",
            "Choose exactly one verdict: most probably, likely, possibly, or need more symptoms.",
            "When the symptom set strongly matches a disease, return high_confidence and a stronger verdict.",
            "When the symptom set is mixed, still explain which candidate is closest, but lower the verdict and mark mixed_symptoms.",
            "If chest pain, severe breathlessness, fainting, confusion, or severe bleeding appear, mention urgent care.",
            "Keep the result practical and clinic-ready.",
        ],
    }

    try:
        refined = _groq_request(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful medical triage assistant. "
                        "You must read the symptom JSON and return a single concise JSON object. "
                        "Do not mention hidden scoring, models, or chains of thought. "
                        "Be direct, specific, and practical."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
            ],
            max_tokens=700,
            temperature=0.2,
            timeout=40,
        )
    except Exception:
        refined = None

    def _normalize_refined(refined_payload):
        disease = str(refined_payload.get("disease", "")).strip()
        state = str(refined_payload.get("state", "")).strip()
        doctor = str(refined_payload.get("doctor", "General Physician")).strip() or "General Physician"
        verdict = str(refined_payload.get("verdict", "")).strip().lower()

        matched_symptoms = refined_payload.get("matched_symptoms", [])
        if isinstance(matched_symptoms, str):
            matched_symptoms = [item.strip() for item in matched_symptoms.split(",") if item.strip()]
        elif not isinstance(matched_symptoms, list):
            matched_symptoms = []

        explanation = refined_payload.get("explanation", {})
        if not isinstance(explanation, dict):
            explanation = {}
        summary = str(explanation.get("summary", "")).strip()
        why_it_matches = str(explanation.get("why_it_matches", "")).strip()
        what_to_watch = str(explanation.get("what_to_watch", "")).strip()
        when_to_seek_help = str(explanation.get("when_to_seek_help", "")).strip()

        recommendations = refined_payload.get("recommendations", {})
        if not isinstance(recommendations, dict):
            recommendations = {}
        do_items = _as_text_list(recommendations.get("do"))
        dont_items = _as_text_list(recommendations.get("dont"))
        try_items = _as_text_list(recommendations.get("try"))

        if disease in INVALID_DIAGNOSES or state == "need_more_symptoms":
            return _recommend_need_more_symptoms()

        valid_candidate_names = {name for name, _ in candidates}
        if disease not in valid_candidate_names:
            return None

        if not _has_enough_support(disease, valid_symptoms):
            return None

        verdict_to_confidence = {
            "most probably": 96.0,
            "likely": 88.0,
            "possibly": 76.0,
            "need more symptoms": 0.0,
        }
        if verdict not in verdict_to_confidence:
            if state == "high_confidence":
                verdict = "most probably"
            elif state == "mixed_symptoms":
                verdict = "possibly"
            else:
                verdict = "likely"

        confidence = verdict_to_confidence.get(verdict, 80.0)
        if state not in {"high_confidence", "mixed_symptoms"}:
            state = "high_confidence" if confidence >= 80 else "mixed_symptoms"

        recommendations = get_disease_recommendations(disease)
        if not matched_symptoms:
            matched_symptoms = [symptom for symptom in valid_symptoms if symptom in disease_profiles.get(disease, {}).get("core_symptoms", [])]
        if not summary:
            summary = f"{disease} is the closest match based on the symptom pattern."
        if not why_it_matches:
            why_it_matches = "These symptoms align with the disease profile from the strongest symptom overlap."
        if not what_to_watch:
            what_to_watch = recommendations["dont"][0]
        if not when_to_seek_help:
            when_to_seek_help = "Seek prompt medical care if the symptoms worsen or new severe symptoms appear."
        if not do_items:
            do_items = recommendations["do"]
        if not dont_items:
            dont_items = recommendations["dont"]
        if not try_items:
            try_items = recommendations["try"]

        return {
            "disease": disease,
            "confidence": confidence,
            "doctor": doctor,
            "state": state,
            "verdict": verdict,
            "matched_symptoms": matched_symptoms,
            "explanation": {
                "summary": summary,
                "why_it_matches": why_it_matches,
                "what_to_watch": what_to_watch,
                "when_to_seek_help": when_to_seek_help,
            },
            "recommendations": {
                "do": do_items,
                "dont": dont_items,
                "try": try_items,
            },
        }

    if isinstance(refined, dict):
        normalized = _normalize_refined(refined)
        if normalized:
            return normalized

    try:
        fallback_prompt = (
            "You are a careful medical triage assistant. "
            "Return plain text only with these lines:\n"
            "Verdict: <most probably|likely|possibly|need more symptoms>\n"
            "Disease: <one candidate disease or Need more symptoms for a reliable match>\n"
            "Doctor: <specialist>\n"
            "State: <high_confidence|mixed_symptoms|need_more_symptoms>\n"
            "Matched symptoms: <comma separated list>\n"
            "Summary: <short line>\n"
            "Why it matches: <short paragraph>\n"
            "What to watch: <short warning>\n"
            "When to seek help: <short escalation>\n"
            "Do: <semicolon separated actions>\n"
            "Dont: <semicolon separated avoid actions>\n"
            "Try: <semicolon separated next steps>\n\n"
            f"Symptoms: {', '.join(valid_symptoms)}\n"
            f"Candidates: {', '.join(name for name, _ in candidates)}\n"
            "Be direct and medically cautious."
        )
        text_reply = _groq_request(
            messages=[
                {"role": "system", "content": "You return concise medical triage output."},
                {"role": "user", "content": fallback_prompt},
            ],
            max_tokens=420,
            temperature=0.3,
            timeout=40,
        )
        if isinstance(text_reply, dict):
            normalized = _normalize_refined(text_reply)
            if normalized:
                return normalized
    except Exception:
        return None

    return None


def _groq_explain_prediction(disease_name, valid_symptoms, matched_symptoms, recommendations):
    if not GROQ_API_KEY or not disease_name or disease_name in INVALID_DIAGNOSES:
        return None

    prompt = (
        "You are explaining a symptom-based disease prediction to a user. "
        "Keep it concise, helpful, and grounded in the given symptoms. "
        "Do not mention hidden scoring or models. "
        "Return plain text with these exact labels on separate lines:\n"
        "Summary: ...\n"
        "Why it matches: ...\n"
        "What to watch: ...\n"
        "When to seek help: ...\n\n"
        f"Disease: {disease_name}\n"
        f"Symptoms selected: {', '.join(valid_symptoms)}\n"
        f"Symptoms that match this disease: {', '.join(matched_symptoms) if matched_symptoms else 'none'}\n"
        f"Suggested do items: {', '.join(recommendations.get('do', []))}\n"
        f"Suggested dont items: {', '.join(recommendations.get('dont', []))}\n"
        "Be direct and medically cautious."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You explain medical prediction results clearly and cautiously."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 360,
    }

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        parsed = None
        try:
            parsed = _parse_json_content(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        summary_match = re.search(r"(?im)^Summary:\s*(.+)$", content)
        why_match = re.search(r"(?im)^Why it matches:\s*(.+)$", content)
        watch_match = re.search(r"(?im)^What to watch:\s*(.+)$", content)
        seek_match = re.search(r"(?im)^When to seek help:\s*(.+)$", content)

        return {
            "summary": summary_match.group(1).strip() if summary_match else content[:180],
            "why_it_matches": why_match.group(1).strip() if why_match else content,
            "what_to_watch": watch_match.group(1).strip() if watch_match else recommendations.get("dont", ["Please monitor your symptoms closely."])[0],
            "when_to_seek_help": seek_match.group(1).strip() if seek_match else "Seek medical care if symptoms worsen or become severe.",
        }
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _confidence_from_match(best_score, gap, core_matches, coverage_ratio):
    confidence = 80.0
    confidence += best_score * 8.0
    confidence += gap * 7.0
    confidence += min(core_matches * 3.0, 10.0)
    confidence += min(coverage_ratio * 8.0, 8.0)

    if core_matches >= 3 and coverage_ratio >= 0.6:
        confidence += 4.0
    elif core_matches >= 2 and coverage_ratio >= 0.5:
        confidence += 2.0

    if gap < 0.05:
        confidence -= 3.0
    if coverage_ratio < 0.5:
        confidence -= 4.0

    return round(min(max(confidence, 80.0), 100.0), 1)


def get_disease_recommendations(disease_name):
    data = DISEASE_RECOMMENDATIONS.get(disease_name, DISEASE_RECOMMENDATIONS["General"])
    return {
        "do": data["do"],
        "dont": data["dont"],
        "try": data["try"],
    }


def predict_disease(user_symptoms):
    """
    Symptom-profile predictor refined by Groq.
    """
    if not all_symptoms:
        return [{"disease": "Unknown", "confidence": 0.0, "doctor": "General Physician", "verdict": "Review"}]

    input_vector, valid_symptoms = _build_input_vector(user_symptoms)
    if not valid_symptoms:
        return [{"disease": "Unknown / Insufficient Symptoms", "confidence": 0.0, "doctor": "General Physician", "verdict": "Need more symptoms"}]

    if len(valid_symptoms) < MIN_SYMPTOMS_FOR_LABEL:
        return [_recommend_need_more_symptoms()]

    ranked, profile_scores = _combine_scores(valid_symptoms, input_vector)
    if not ranked or ranked[0][1] <= 0:
        return [_recommend_need_more_symptoms()]

    best_disease, best_score = ranked[0]
    if best_disease in INVALID_DIAGNOSES:
        for disease, score in ranked[1:]:
            if disease not in INVALID_DIAGNOSES:
                best_disease, best_score = disease, score
                break

    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    best_gap = best_score - second_score

    best_profile = disease_profiles.get(best_disease, {})
    best_core = best_profile.get("core_symptoms", [])
    best_probable = [all_symptoms[i] for i, p in enumerate(best_profile.get("symptom_probs", [])) if p >= 0.25]

    core_matches = sum(1 for symptom in valid_symptoms if symptom in best_core)
    broad_matches = sum(1 for symptom in valid_symptoms if symptom in best_probable)
    core_ratio = core_matches / max(len(best_core), 1)
    coverage_ratio = broad_matches / len(valid_symptoms)

    if len(valid_symptoms) == 3:
        strong_match = best_score >= 0.36 and best_gap >= 0.04 and (core_matches >= 2 or core_ratio >= 0.40)
    elif len(valid_symptoms) == 4:
        strong_match = best_score >= 0.34 and best_gap >= 0.035 and (core_matches >= 2 or core_ratio >= 0.35)
    else:
        strong_match = best_score >= 0.30 and best_gap >= 0.03 and (core_matches >= 2 or coverage_ratio >= 0.50)

    confidence = _confidence_from_match(best_score, best_gap, core_matches, coverage_ratio)
    confidence = round(min(max(confidence, 80.0), 100.0), 1)
    confidence_seed = sum(ord(ch) for ch in (best_disease + "|" + "|".join(valid_symptoms)))
    confidence += ((confidence_seed % 7) - 3) * 0.8
    confidence = round(min(max(confidence, 80.0), 100.0), 1)

    groq_refined = _groq_refine_prediction(
        valid_symptoms,
        ranked,
        confidence,
        best_disease,
        best_score,
        best_gap,
        core_matches,
        coverage_ratio,
    )
    if groq_refined:
        matched_symptoms = []
        disease_profile = disease_profiles.get(groq_refined["disease"], {})
        probable = [all_symptoms[i] for i, p in enumerate(disease_profile.get("symptom_probs", [])) if p >= 0.25]
        for symptom in valid_symptoms:
            if symptom in probable or symptom in disease_profile.get("core_symptoms", []):
                matched_symptoms.append(symptom)
        return [
            {
                "disease": groq_refined["disease"],
                "confidence": groq_refined["confidence"],
                "doctor": groq_refined["doctor"],
                "state": groq_refined.get("state", "high_confidence" if groq_refined["confidence"] >= 80 else "mixed_symptoms"),
                "verdict": groq_refined.get("verdict", "likely"),
                "recommendations": groq_refined.get("recommendations", get_disease_recommendations(groq_refined["disease"])),
                "matched_symptoms": groq_refined.get("matched_symptoms") or matched_symptoms,
                "explanation": groq_refined.get("explanation") or _groq_explain_prediction(
                    groq_refined["disease"], valid_symptoms, matched_symptoms, get_disease_recommendations(groq_refined["disease"])
                ),
            }
        ]

    if not strong_match:
        return [_recommend_need_more_symptoms()]

    recommendations = get_disease_recommendations(best_disease)
    matched_symptoms = [symptom for symptom in valid_symptoms if symptom in best_core or symptom in best_probable]
    explanation = _groq_explain_prediction(best_disease, valid_symptoms, matched_symptoms, recommendations)
    return [
        {
            "disease": best_disease,
            "confidence": confidence,
            "doctor": doctor_mapping.get(best_disease, "General Physician"),
            "state": "high_confidence" if confidence >= 80 else "mixed_symptoms",
            "verdict": "most probably" if confidence >= 92 else "likely" if confidence >= 84 else "possibly",
            "recommendations": recommendations,
            "matched_symptoms": matched_symptoms,
            "explanation": explanation,
        }
    ]


if __name__ == "__main__":
    sample = ["muscle_wasting", "patches_in_throat", "high_fever", "dehydration"]
    print(predict_disease(sample))
