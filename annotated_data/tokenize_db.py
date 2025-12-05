import unicodedata

import spacy
import srsly

# Load a blank English model
nlp = spacy.blank("en")

# Input and output file paths
input_path = "db_20251201.jsonl"
output_path = "annotated_data/db_20251201_tokenised.jsonl"

# Read all Prodigy-style examples
examples = srsly.read_jsonl(input_path)
output_data = []


for example in examples:
    # ensure it's a dict (JSONL can contain blank lines)
    if not isinstance(example, dict):
        continue

    # Skip examples from this annotator
    # if example.get("_annotator_id") == "handover_db-user1":
    #     continue

    text = example.get("text", "")
    if not text:
        continue

    # Normalize text
    text = unicodedata.normalize("NFC", text)

    # Tokenize text with spaCy
    doc = nlp(text)
    tokens = [
        {
            "text": token.text,
            "start": token.idx,
            "end": token.idx + len(token.text),
            "id": i,
            "ws": token.whitespace_ != "",
        }
        for i, token in enumerate(doc)
    ]

    # ✅ Copy all existing keys and add tokens
    tokenised_example = dict(example)
    tokenised_example["tokens"] = tokens
    # Change view ID
    tokenised_example["_view_id"] = "choice"
    tokenised_example["options"] = [
        {
            "id": "introduction_clinicians",
            "text": "Introduction of clinicians involved in handover",
        },
        {
            "id": "introduction_patients",
            "text": "Introduction of clinicians involved in handover to patient/carer",
        },
        {
            "id": "invitation",
            "text": "Invitation for patient/carer to participate in handover",
        },
        {"id": "id_check", "text": "ID check of 3 patient identifiers"},
        {
            "id": "sit_diagnosis",
            "text": "Primary diagnosis | reason for admission",
        },
        {"id": "sit_events", "text": "Significant events or complications"},
        {
            "id": "sit_status",
            "text": "Current status (awaiting tests/procedures, on interim orders/plan)",
        },
        {
            "id": "bg_history",
            "text": "Relevant clinical and social history | comorbidities",
        },
        {"id": "bg_fall_risk", "text": "Alerts - falls risk"},
        {"id": "bg_pi_risk", "text": "Alerts - pressure injury risk"},
        {"id": "bg_allergies", "text": "Alerts - allergies"},
        {"id": "bg_acp", "text": "Advanced care planning"},
        {
            "id": "assess_obs",
            "text": "Observations | Q-ADDS | recent escalations",
        },
        {"id": "assess_pain", "text": "Pain management"},
        {"id": "assess_devices", "text": "Devices | lines | vascular access"},
        {"id": "assess_monitoring", "text": "Critical monitoring | alarms"},
        {"id": "assess_nutrition", "text": "Nutrition | restrictions"},
        {"id": "assess_fluid_balance", "text": "Fluid balance | restrictions"},
        {"id": "assess_infusions", "text": "Infusions"},
        {
            "id": "assess_medications",
            "text": "Medication chart | flag high risk meds",
        },
        {"id": "assess_pathology", "text": "Pathology"},
        {"id": "assess_mobility", "text": "Mobility | aids"},
        {
            "id": "assess_skin_integrity",
            "text": "Skin integrity | interventions",
        },
        {"id": "rec_discharge_plan", "text": "Discharge plan"},
        {"id": "rec_actions", "text": "Critical actions required"},
        {"id": "rec_plan", "text": "Care plan/pathway actions to follow up"},
        {
            "id": "rec_patient_goals",
            "text": "Asked patient/carer about goals and preferences",
        },
    ]

    output_data.append(tokenised_example)


# ✅ Write to JSONL file (Prodigy format)
srsly.write_jsonl(output_path, output_data)
print(f"✅ Tokenised data written to {output_path}")
