"""Interactive terminal for Oracle — diagnostic pattern completion."""

from __future__ import annotations

import json
import os

from .engine import Oracle
from .symptoms import SYMPTOMS, CATEGORIES, INDEX_TO_SYMPTOM

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def severity_bar(val: float) -> str:
    filled = int(val * 15)
    bar = "█" * filled + "░" * (15 - filled)
    if val > 0.7:
        return f"{RED}{bar}{RESET}"
    if val > 0.4:
        return f"{YELLOW}{bar}{RESET}"
    return f"{GREEN}{bar}{RESET}"


def load_sample_data(oracle: Oracle) -> dict:
    """Load and ingest sample patient data."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "sample_data", "patients.json")
    if not os.path.exists(path):
        print(f"{RED}  sample data not found{RESET}")
        return {}

    with open(path) as f:
        data = json.load(f)

    patients = data.get("patients", [])
    profiles = [p["symptoms"] for p in patients]
    oracle.store_patients(profiles)
    print(f"{GREEN}  Loaded {len(profiles)} patient profiles into EAM{RESET}")
    return data.get("conditions", {})


def cmd_diagnose(oracle: Oracle, symptom_str: str):
    """Diagnose from a comma-separated symptom list."""
    # Parse: "fever:0.8, headache:0.9, cough" → dict
    observed = {}
    for part in symptom_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, val = part.split(":", 1)
            name = name.strip().replace(" ", "_")
            try:
                observed[name] = float(val.strip())
            except ValueError:
                observed[name] = 0.7
        else:
            name = part.strip().replace(" ", "_")
            observed[name] = 0.7  # default severity

    # Validate
    unknown = [s for s in observed if s not in SYMPTOMS]
    if unknown:
        print(f"{YELLOW}  unknown symptoms: {', '.join(unknown)}{RESET}")
        print(f"{DIM}  available: {', '.join(sorted(SYMPTOMS.keys()))}{RESET}\n")
        return

    if not observed:
        print(f"{DIM}  no symptoms provided{RESET}\n")
        return

    result = oracle.diagnose(observed)

    # Display
    print(f"\n{BOLD}Diagnosis — pattern completion{RESET}")
    print(f"{DIM}fidelity: {result.fidelity:.3f} "
          f"(how strongly SDM recognizes this symptom pattern){RESET}\n")

    # Observed symptoms (confirmed in reconstruction)
    print(f"  {BOLD}Observed:{RESET}")
    for name, sev in sorted(observed.items(), key=lambda x: x[1], reverse=True):
        recon_val = result.predicted.get(name, 0.0)
        match = f"{GREEN}✓{RESET}" if recon_val > 0.05 else f"{RED}✗{RESET}"
        print(f"    {match} {name:25s} {severity_bar(sev)} {sev:.1f}")

    # Predicted symptoms (NEW — not observed)
    new_preds = result.new_predictions
    if new_preds:
        sorted_preds = sorted(new_preds.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  {BOLD}{YELLOW}Predicted (check these):{RESET}")
        for name, sev in sorted_preds[:10]:
            print(f"    → {name:25s} {severity_bar(sev)} {sev:.3f}")
    else:
        print(f"\n  {DIM}no additional symptoms predicted{RESET}")

    print()


def cmd_differential(oracle: Oracle, symptom_str: str,
                     conditions: dict):
    """Show differential diagnosis — which condition matches best."""
    observed = {}
    for part in symptom_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, val = part.split(":", 1)
            observed[name.strip().replace(" ", "_")] = float(val.strip())
        else:
            observed[part.strip().replace(" ", "_")] = 0.7

    if not observed or not conditions:
        print(f"{DIM}  need symptoms and condition data{RESET}\n")
        return

    condition_profiles = {name: info["typical_profile"]
                          for name, info in conditions.items()}
    rankings = oracle.compare_conditions(observed, condition_profiles)

    print(f"\n{BOLD}Differential Diagnosis{RESET}")
    print(f"{DIM}Observed: {', '.join(observed.keys())}{RESET}\n")

    for i, (name, score) in enumerate(rankings):
        desc = conditions.get(name, {}).get("description", "")
        bar_len = int(max(0, score) * 30)
        bar = "█" * bar_len
        color = GREEN if i == 0 else YELLOW if i < 3 else DIM
        marker = "►" if i == 0 else " "
        print(f"  {marker} {color}{name:25s} {score:.3f}  {bar}{RESET}")
        if desc and i < 3:
            print(f"    {DIM}{desc}{RESET}")

    print()


def cmd_next(oracle: Oracle, symptom_str: str):
    """Suggest what to check next."""
    observed = {}
    for part in symptom_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, val = part.split(":", 1)
            observed[name.strip().replace(" ", "_")] = float(val.strip())
        else:
            observed[part.strip().replace(" ", "_")] = 0.7

    if not observed:
        print(f"{DIM}  no symptoms provided{RESET}\n")
        return

    suggestions = oracle.differential(observed, top_k=8)

    print(f"\n{BOLD}Suggested next checks:{RESET}")
    print(f"{DIM}Based on SDM pattern completion — these symptoms are predicted{RESET}")
    print(f"{DIM}but not yet observed. Check them to confirm or rule out.{RESET}\n")

    for name, confidence in suggestions:
        bar = severity_bar(confidence)
        print(f"    → {name:25s} {bar} {confidence:.3f}")

    print()


def cmd_symptoms(category: str | None = None):
    """List available symptoms."""
    if category:
        cat = category.capitalize()
        if cat in CATEGORIES:
            indices = CATEGORIES[cat]
            print(f"\n{BOLD}{cat} symptoms:{RESET}")
            for idx in indices:
                print(f"  {INDEX_TO_SYMPTOM[idx]}")
            print()
            return

    print(f"\n{BOLD}Symptom categories:{RESET}")
    for cat, indices in CATEGORIES.items():
        names = [INDEX_TO_SYMPTOM[i] for i in indices]
        print(f"  {CYAN}{cat:20s}{RESET} {DIM}{', '.join(names)}{RESET}")
    print()


def run(oracle: Oracle):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Oracle{RESET} — diagnostic pattern completion")
    print(f"{DIM}Store patient profiles → query with partial symptoms → SDM completes the pattern{RESET}")
    print(f"{DIM}Commands:{RESET}")
    print(f"{DIM}  diagnose <symptoms>    Pattern completion (e.g. fever:0.8, headache, cough:0.5){RESET}")
    print(f"{DIM}  diff <symptoms>        Differential diagnosis against known conditions{RESET}")
    print(f"{DIM}  next <symptoms>        Suggest what to check next{RESET}")
    print(f"{DIM}  symptoms [category]    List available symptoms{RESET}")
    print(f"{DIM}  stats                  SDM statistics{RESET}")
    print(f"{DIM}  quit                   Exit{RESET}\n")

    # Auto-load sample data
    print(f"{DIM}Loading sample patient data...{RESET}")
    conditions = load_sample_data(oracle)
    print()

    while True:
        try:
            user_input = input(f"{CYAN}oracle>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        parts = user_input.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if cmd in ("diagnose", "d"):
            if not args:
                print(f"{DIM}  usage: diagnose fever:0.8, headache, stiff_neck:1.0{RESET}\n")
                continue
            try:
                cmd_diagnose(oracle, args)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd in ("diff", "differential"):
            if not args:
                print(f"{DIM}  usage: diff fever:0.8, headache, stiff_neck{RESET}\n")
                continue
            try:
                cmd_differential(oracle, args, conditions)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "next":
            if not args:
                print(f"{DIM}  usage: next fever, headache{RESET}\n")
                continue
            try:
                cmd_next(oracle, args)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "symptoms":
            cmd_symptoms(args.strip() if args else None)
            continue

        if cmd == "stats":
            try:
                stats = oracle.heather.stats()
                print(f"\n{DIM}  Patient profiles stored: {oracle.profiles_stored}")
                print(f"  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        print(f"{DIM}  unknown command. try: diagnose, diff, next, symptoms, stats, quit{RESET}\n")
