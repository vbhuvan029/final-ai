import math
import random
from pathlib import Path

import pandas as pd


BASE_DATASET = Path("ml_model/Training.csv")
OUTPUT_DATASET = Path("ml_model/Augmented_Training.csv")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [col for col in df.columns if "Unnamed" in str(col)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.columns = df.columns.str.strip()
    df["prognosis"] = df["prognosis"].str.strip()
    df = df.dropna(how="all")
    return df


def _build_profiles(df: pd.DataFrame, symptoms: list[str]) -> dict:
    profiles = {}
    for disease, group in df.groupby("prognosis"):
        symptom_probs = group[symptoms].mean().values
        core_symptoms = [symptoms[i] for i, p in enumerate(symptom_probs) if p >= 0.6]
        frequent_symptoms = [symptoms[i] for i, p in enumerate(symptom_probs) if p >= 0.25]
        rare_symptoms = [symptoms[i] for i, p in enumerate(symptom_probs) if 0.05 <= p < 0.25]
        profiles[disease] = {
            "symptom_probs": symptom_probs,
            "core_symptoms": core_symptoms,
            "frequent_symptoms": frequent_symptoms,
            "rare_symptoms": rare_symptoms,
        }
    return profiles


def _mutate_row(base_row: pd.Series, profile: dict, symptoms: list[str], rng: random.Random) -> dict:
    row = base_row.to_dict()
    active = [symptom for symptom in symptoms if int(row.get(symptom, 0)) == 1]
    core = profile["core_symptoms"]
    frequent = profile["frequent_symptoms"]
    rare = profile["rare_symptoms"]

    # Drop a few symptoms to simulate incomplete user input.
    if active:
        drop_limit = max(1, min(4, math.ceil(len(active) * 0.35)))
        drop_count = rng.randint(0, drop_limit)
        protected = set(rng.sample(core, min(len(core), max(1, len(core) // 2)))) if core else set()
        droppable = [symptom for symptom in active if symptom not in protected]
        if droppable and drop_count > 0:
            for symptom in rng.sample(droppable, min(drop_count, len(droppable))):
                row[symptom] = 0

    # Add likely supporting symptoms for the same disease.
    add_pool = [symptom for symptom in frequent if int(row.get(symptom, 0)) == 0]
    if add_pool:
        add_count = rng.randint(0, min(3, len(add_pool)))
        for symptom in rng.sample(add_pool, add_count):
            if rng.random() < 0.65:
                row[symptom] = 1

    # Occasionally add a rare but still plausible symptom.
    rare_pool = [symptom for symptom in rare if int(row.get(symptom, 0)) == 0]
    if rare_pool and rng.random() < 0.35:
        row[rng.choice(rare_pool)] = 1

    # Keep at least two active symptoms when possible.
    active_after = [symptom for symptom in symptoms if int(row.get(symptom, 0)) == 1]
    if len(active_after) < 2:
        restore_pool = core or frequent or active or symptoms
        for symptom in rng.sample(restore_pool, min(2, len(restore_pool))):
            row[symptom] = 1

    return row


def augment_dataset(
    input_file: str | Path = BASE_DATASET,
    output_file: str | Path = OUTPUT_DATASET,
    target_total_rows: int = 15000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    input_file = Path(input_file)
    output_file = Path(output_file)

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    df = _clean_frame(df)

    symptoms = [col for col in df.columns if col != "prognosis"]
    diseases = sorted(df["prognosis"].unique())
    profiles = _build_profiles(df, symptoms)

    current_rows = len(df)
    target_per_disease = max(
        math.ceil(target_total_rows / max(len(diseases), 1)),
        max(df["prognosis"].value_counts().max(), 1),
    )

    print(f"Base rows: {current_rows}")
    print(f"Diseases: {len(diseases)} | Symptoms: {len(symptoms)}")
    print(f"Target per disease: {target_per_disease}")

    augmented_rows = []
    for disease in diseases:
        disease_group = df[df["prognosis"] == disease]
        disease_rows = disease_group.to_dict("records")
        required = max(0, target_per_disease - len(disease_group))

        if not disease_rows or required == 0:
            continue

        for i in range(required):
            template = pd.Series(disease_rows[i % len(disease_rows)])
            synthetic_row = _mutate_row(template, profiles[disease], symptoms, rng)
            synthetic_row["prognosis"] = disease
            augmented_rows.append(synthetic_row)

    augmented_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)

    print(f"Augmented dataset saved to {output_file}")
    print(f"Original rows: {len(df)}")
    print(f"Added synthetic rows: {len(augmented_df)}")
    print(f"Final rows: {len(final_df)}")
    return final_df


if __name__ == "__main__":
    augment_dataset()
