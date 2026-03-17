"""Symptom/condition encoding for diagnostic pattern completion.

Each patient profile is a vector where dimensions represent symptoms.
The value encodes severity: 0.0 = absent, 0.5 = mild, 1.0 = severe.

When a partial profile (some symptoms unknown) is queried against the EAM,
the reconstruction COMPLETES the pattern — predicting the likely values
of unobserved symptoms based on all stored patient profiles.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────
# Symptom catalog — 64 symptoms across multiple body systems
# Each symptom has an index (its dimension in the vector)
# ──────────────────────────────────────────────────────────────────

SYMPTOMS: dict[str, int] = {
    # General (0-7)
    "fever": 0,
    "fatigue": 1,
    "weight_loss": 2,
    "night_sweats": 3,
    "chills": 4,
    "malaise": 5,
    "appetite_loss": 6,
    "dehydration": 7,

    # Head/Neuro (8-15)
    "headache": 8,
    "dizziness": 9,
    "confusion": 10,
    "seizures": 11,
    "light_sensitivity": 12,
    "blurred_vision": 13,
    "stiff_neck": 14,
    "numbness": 15,

    # Respiratory (16-23)
    "cough": 16,
    "shortness_breath": 17,
    "chest_pain": 18,
    "wheezing": 19,
    "sore_throat": 20,
    "runny_nose": 21,
    "sneezing": 22,
    "bloody_sputum": 23,

    # Gastrointestinal (24-31)
    "nausea": 24,
    "vomiting": 25,
    "diarrhea": 26,
    "abdominal_pain": 27,
    "bloating": 28,
    "constipation": 29,
    "blood_stool": 30,
    "acid_reflux": 31,

    # Musculoskeletal (32-39)
    "joint_pain": 32,
    "muscle_ache": 33,
    "back_pain": 34,
    "swelling": 35,
    "stiffness": 36,
    "weakness": 37,
    "cramps": 38,
    "limited_mobility": 39,

    # Skin (40-47)
    "rash": 40,
    "itching": 41,
    "bruising": 42,
    "swollen_lymph": 43,
    "skin_lesions": 44,
    "jaundice": 45,
    "hair_loss": 46,
    "dry_skin": 47,

    # Cardiovascular (48-55)
    "rapid_heartbeat": 48,
    "irregular_heartbeat": 49,
    "high_blood_pressure": 50,
    "low_blood_pressure": 51,
    "swollen_legs": 52,
    "chest_tightness": 53,
    "fainting": 54,
    "cold_extremities": 55,

    # Mental/Other (56-63)
    "anxiety": 56,
    "depression": 57,
    "insomnia": 58,
    "irritability": 59,
    "frequent_urination": 60,
    "painful_urination": 61,
    "excessive_thirst": 62,
    "excessive_hunger": 63,
}

DIMENSION = len(SYMPTOMS)  # 64

# Reverse lookup
INDEX_TO_SYMPTOM = {v: k for k, v in SYMPTOMS.items()}

# Symptom categories for display
CATEGORIES = {
    "General": list(range(0, 8)),
    "Neurological": list(range(8, 16)),
    "Respiratory": list(range(16, 24)),
    "Gastrointestinal": list(range(24, 32)),
    "Musculoskeletal": list(range(32, 40)),
    "Skin": list(range(40, 48)),
    "Cardiovascular": list(range(48, 56)),
    "Mental/Other": list(range(56, 64)),
}


def encode_profile(symptoms: dict[str, float]) -> list[float]:
    """Encode a symptom profile as a 64-dim vector.
    symptoms: {symptom_name: severity} where severity is 0.0 to 1.0.
    Unknown/unobserved symptoms are set to 0.0."""
    import numpy as np
    vec = np.zeros(DIMENSION, dtype=np.float64)
    for name, severity in symptoms.items():
        if name in SYMPTOMS:
            vec[SYMPTOMS[name]] = severity
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def decode_profile(vec: list[float], threshold: float = 0.05) -> dict[str, float]:
    """Decode a vector back to symptom names and severities.
    Returns symptoms above threshold."""
    import numpy as np
    v = np.array(vec, dtype=np.float64)
    # Denormalize — scale so max is ~1.0
    max_val = np.max(np.abs(v))
    if max_val > 0:
        v = v / max_val
    result = {}
    for name, idx in SYMPTOMS.items():
        if v[idx] > threshold:
            result[name] = round(float(v[idx]), 3)
    return result
