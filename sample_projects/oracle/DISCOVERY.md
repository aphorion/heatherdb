# Discovery: Diagnostic Pattern Completion — SDM Predicts Missing Symptoms

**Date:** 2026-02-13
**Project:** Oracle (HeatherDB sample project)
**Finding:** When patient symptom profiles are stored in SDM, querying with partial symptoms reconstructs the COMPLETE expected profile — predicting symptoms the patient hasn't reported yet. The reconstruction is a composite of all similar patients, not any single stored case.

## Setup

- 64-dimensional vectors: one dimension per symptom (8 categories × 8 symptoms)
- Symptom values: 0.0 (absent) to 1.0 (severe)
- Patient profiles stored in SDM (40 patients across 10 conditions)
- Query: encode partial symptoms → SDM reconstructs complete profile
- New predictions = symptoms in reconstruction not present in query

## Key Findings

### 1. Pattern Completion as Diagnosis

Given partial symptoms like `fever:0.8, headache:0.7, stiff_neck:0.9`:
- **Vector DB**: returns the most similar stored patient profile
- **SDM**: reconstructs a composite pattern from ALL patients with similar symptoms

The reconstruction predicts what other symptoms typically accompany this combination. For meningitis-like symptoms, it might predict `light_sensitivity`, `confusion`, `nausea` — symptoms the clinician should check for.

### 2. Composite vs Single Match

The EAM reconstruction isn't any single patient — it's the statistical consensus of all patients whose profiles overlap with the query. This means:
- Common symptom co-occurrences are amplified (strong predictions)
- Rare/idiosyncratic symptoms are suppressed (weak or absent)
- The prediction reflects population-level patterns, not individual cases

### 3. Differential Diagnosis from Fidelity

Fidelity (cosine of query vs reconstruction) indicates how well the symptoms fit known patterns:
- **High fidelity**: clear pattern match — symptoms strongly resemble a known condition
- **Low fidelity**: unusual combination — may be a rare condition, atypical presentation, or multiple conditions

This gives clinicians a confidence signal alongside the prediction.

### 4. "What to Check Next"

The `differential()` function ranks predicted symptoms by severity that haven't been observed yet. This answers: "given what you've told me, what should we check next?" — prioritizing the most diagnostically informative tests.

## Why This Matters

| | Lookup/Database | ML Classifier | Oracle (SDM) |
|---|---|---|---|
| Partial input | Fails (needs full record) | Needs training data | Pattern completes naturally |
| Output | Matching record(s) | Condition label + probability | Complete symptom profile |
| Explains prediction | Shows similar patient | Black box | Shows predicted symptoms individually |
| Multiple conditions | Returns top matches | Multi-label classification | Reconstruction blends co-morbidities |
| No training | N/A | Needs labeled data | Just store profiles |

The key insight: **SDM treats diagnosis as pattern completion, not classification.** It doesn't ask "which disease is this?" — it asks "what does a complete picture of this patient typically look like?" The missing pieces ARE the diagnosis.

## Parameters
- Dimension: 64 (one per symptom, 8 categories × 8 symptoms)
- 40 patient profiles across 10 conditions
- Conditions: flu, meningitis, diabetes type 2, pneumonia, rheumatoid arthritis, heart failure, gastroenteritis, migraine, depression, UTI
- SDM: k=20 activation, Hopfield iterative read, beta=5.0
