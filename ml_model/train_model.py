import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from augment_dataset import augment_dataset


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
BASE_DATASET = BASE_DIR / "Training.csv"
AUGMENTED_DATASET = BASE_DIR / "Augmented_Training.csv"
SUMMARY_PATH = BASE_DIR / "training_summary.json"
MODEL_PATHS = {
    "random_forest": BASE_DIR / "random_forest_model.pkl",
    "extra_trees": BASE_DIR / "extra_trees_model.pkl",
    "gradient_boosting": BASE_DIR / "gradient_boosting_model.pkl",
    "logistic_regression": BASE_DIR / "logistic_regression_model.pkl",
    "naive_bayes": BASE_DIR / "naive_bayes_model.pkl",
}
SYMPTOMS_PATH = BASE_DIR / "symptoms.json"
DOCTOR_MAPPING_PATH = BASE_DIR / "doctor_mapping.json"
PROFILES_PATH = BASE_DIR / "disease_profiles.json"
INVALID_DIAGNOSES = {"Unknown", "Unknown / Insufficient Symptoms", "Need more symptoms for a reliable match"}


def _clean_frame(df):
    df = df.copy()
    drop_cols = [col for col in df.columns if "Unnamed" in str(col)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.columns = df.columns.str.strip()
    df["prognosis"] = df["prognosis"].str.strip()
    return df.dropna(how="all")


def _load_dataset():
    if AUGMENTED_DATASET.exists():
        print(f"Loading augmented dataset: {AUGMENTED_DATASET}")
        df = pd.read_csv(AUGMENTED_DATASET)
    else:
        print("Augmented dataset not found. Generating one now...")
        df = augment_dataset(BASE_DATASET, AUGMENTED_DATASET)
    df = _clean_frame(df)
    before = len(df)
    df = df[~df["prognosis"].isin(INVALID_DIAGNOSES)].copy()
    if len(df) != before:
        print(f"  Removed {before - len(df)} invalid diagnosis rows from augmented data")
        df.to_csv(AUGMENTED_DATASET, index=False)
    return df


def _build_profiles(df, symptoms):
    profiles = {}
    for disease, group in df.groupby("prognosis"):
        symptom_probs = group[symptoms].mean().values
        core_mask = symptom_probs >= 0.6
        profiles[disease] = {
            "symptom_probs": symptom_probs,
            "core_symptoms": [symptoms[i] for i in range(len(symptoms)) if core_mask[i]],
            "num_core": int(core_mask.sum()),
        }
    return profiles


def _score_profiles(valid_symptoms, symptoms, profiles):
    scores = {}
    for disease, profile in profiles.items():
        probable = [symptoms[i] for i, p in enumerate(profile["symptom_probs"]) if p >= 0.25]
        core = profile["core_symptoms"]
        if not probable:
            continue

        broad_matches = sum(1 for s in valid_symptoms if s in probable)
        core_matches = sum(1 for s in valid_symptoms if s in core)
        if broad_matches == 0:
            continue

        coverage = broad_matches / len(valid_symptoms)
        core_coverage = core_matches / max(len(core), 1)
        specificity = broad_matches / len(probable)
        core_density = core_matches / len(valid_symptoms)
        scores[disease] = (coverage * 0.35) + (core_coverage * 0.35) + (specificity * 0.10) + (core_density * 0.20)
    return scores


def _save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    print("=" * 60)
    print("  AI Disease Prediction - Ensemble Training Pipeline")
    print("=" * 60)

    print("\n[1/7] Loading dataset...")
    df = _load_dataset()
    symptoms = [col for col in df.columns if col != "prognosis"]
    diseases = sorted(df["prognosis"].unique())
    print(f"  Total samples: {len(df)}")
    print(f"  Diseases: {len(diseases)} | Symptoms: {len(symptoms)}")

    print("\n[2/7] Building disease-symptom profiles...")
    disease_profiles = _build_profiles(df, symptoms)
    print(f"  Built profiles for {len(disease_profiles)} diseases")

    print("\n[3/7] Splitting data...")
    X = df[symptoms].values
    y = df["prognosis"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "naive_bayes": BernoulliNB(alpha=0.75),
    }

    accuracies = {}
    fitted_models = {}

    print("\n[4/7] Training ensemble models...")
    for name, model in models.items():
        print(f"  Training {name.replace('_', ' ').title()}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred) * 100
        accuracies[name] = round(acc, 2)
        print(f"    Test Accuracy: {acc:.2f}%")

    print("\n[5/7] Building profile-based sanity checks...")
    test_cases = [
        (["high_fever", "constipation"], "Typhoid"),
        (["itching", "skin_rash", "nodal_skin_eruptions"], "Fungal infection"),
        (["continuous_sneezing", "shivering", "chills"], "Allergy"),
        (["headache", "chest_pain", "dizziness"], "Hypertension"),
        (["cough", "high_fever", "breathlessness"], "Pneumonia"),
        (["joint_pain", "vomiting", "fatigue", "high_fever", "sweating", "headache"], "Malaria"),
        (["muscle_wasting", "patches_in_throat", "high_fever", "dehydration"], "AIDS"),
    ]

    def predict_preview(user_symptoms):
        vector = np.zeros(len(symptoms))
        valid_symptoms = []
        for symptom in user_symptoms:
            clean = symptom.strip()
            if clean in symptoms:
                vector[symptoms.index(clean)] = 1
                valid_symptoms.append(clean)

        if not valid_symptoms:
            return "Unknown", 0.0

        profile_scores = _score_profiles(valid_symptoms, symptoms, disease_profiles)
        combined = {disease: profile_scores.get(disease, 0.0) * 0.6 for disease in diseases}
        model_weights = {
            "random_forest": 1.1,
            "extra_trees": 1.1,
            "gradient_boosting": 0.9,
            "logistic_regression": 1.0,
            "naive_bayes": 0.8,
        }

        for name, model in fitted_models.items():
            probs = model.predict_proba([vector])[0]
            weight = model_weights[name]
            for idx, disease in enumerate(model.classes_):
                combined[disease] = combined.get(disease, 0.0) + (probs[idx] * weight)

        ranked = sorted(
            ((disease, score) for disease, score in combined.items() if disease not in INVALID_DIAGNOSES),
            key=lambda item: item[1],
            reverse=True,
        )
        best_disease, best_score = ranked[0]
        confidence = min(max(best_score * 100, 35.0), 98.0)
        return best_disease, confidence

    passed = 0
    for symptoms_test, expected in test_cases:
        predicted, conf = predict_preview(symptoms_test)
        status = "OK" if predicted == expected else "MISS"
        if predicted == expected:
            passed += 1
        print(f"  [{status}] {symptoms_test[:3]} -> {predicted} ({conf:.1f}%) [expected: {expected}]")

    print("\n[6/7] Saving model and metadata...")
    for name, model in fitted_models.items():
        _save_pickle(MODEL_PATHS[name], model)

    with open(SYMPTOMS_PATH, "w") as f:
        json.dump(symptoms, f, indent=2)

    with open(PROFILES_PATH, "w") as f:
        profiles_json = {
            disease: {
                "symptom_probs": profile["symptom_probs"].tolist(),
                "core_symptoms": profile["core_symptoms"],
                "num_core": profile["num_core"],
            }
            for disease, profile in disease_profiles.items()
        }
        json.dump(profiles_json, f, indent=2)

    doctor_mapping = {
        "Fungal infection": "Dermatologist",
        "Allergy": "Allergist",
        "GERD": "Gastroenterologist",
        "Chronic cholestasis": "Hepatologist",
        "Drug Reaction": "Allergist",
        "Peptic ulcer diseae": "Gastroenterologist",
        "AIDS": "Infectious Disease Specialist",
        "Diabetes": "Endocrinologist",
        "Gastroenteritis": "Gastroenterologist",
        "Bronchial Asthma": "Pulmonologist",
        "Hypertension": "Cardiologist",
        "Migraine": "Neurologist",
        "Cervical spondylosis": "Orthopedist / Neurologist",
        "Paralysis (brain hemorrhage)": "Neurologist",
        "Jaundice": "Gastroenterologist",
        "Malaria": "Infectious Disease Specialist / General Physician",
        "Chicken pox": "Pediatrician / General Physician",
        "Dengue": "Infectious Disease Specialist / General Physician",
        "Typhoid": "General Physician",
        "hepatitis A": "Hepatologist",
        "Hepatitis B": "Hepatologist",
        "Hepatitis C": "Hepatologist",
        "Hepatitis D": "Hepatologist",
        "Hepatitis E": "Hepatologist",
        "Alcoholic hepatitis": "Hepatologist",
        "Tuberculosis": "Pulmonologist",
        "Common Cold": "General Physician",
        "Pneumonia": "Pulmonologist",
        "Dimorphic hemmorhoids(piles)": "Proctologist",
        "Heart attack": "Cardiologist",
        "Varicose veins": "Vascular Surgeon",
        "Hypothyroidism": "Endocrinologist",
        "Hyperthyroidism": "Endocrinologist",
        "Hypoglycemia": "Endocrinologist",
        "Osteoarthristis": "Rheumatologist",
        "Arthritis": "Rheumatologist",
        "(vertigo) Paroymsal  Positional Vertigo": "ENT Specialist",
        "Acne": "Dermatologist",
        "Urinary tract infection": "Urologist",
        "Psoriasis": "Dermatologist",
        "Impetigo": "Dermatologist",
    }
    with open(DOCTOR_MAPPING_PATH, "w") as f:
        json.dump(doctor_mapping, f, indent=2)

    summary = {
        "model_name": "Ensemble (RF + ExtraTrees + GB + Logistic + NB + Profile)",
        "accuracies": accuracies,
        "total_diseases": len(diseases),
        "total_symptoms": len(symptoms),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "augmented_dataset_rows": len(df),
        "sanity_checks_passed": passed,
        "sanity_checks_total": len(test_cases),
        "diseases": list(diseases),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Accuracies: {accuracies}")
    print(f"  Sanity Checks: {passed}/{len(test_cases)} passed")
    print(f"  Dataset rows: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
