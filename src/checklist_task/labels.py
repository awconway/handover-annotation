from typing import Literal

LabelType = Literal[
    "introduction_clinicians",
    "introduction_patients",
    "invitation",
    "id_check",
    "sit_diagnosis",
    "sit_events",
    "sit_status",
    "bg_history",
    "bg_fall_risk",
    "bg_pi_risk",
    "bg_allergies",
    "bg_acp",
    "assess_obs",
    "assess_pain",
    "assess_devices",
    "assess_monitoring",
    "assess_nutrition",
    "assess_fluid_balance",
    "assess_infusions",
    "assess_medications",
    "assess_pathology",
    "assess_mobility",
    "assess_skin_integrity",
    "rec_discharge_plan",
    "rec_actions",
    "rec_plan",
    "rec_patient_goals",
]

LABEL_DESCRIPTIONS: dict[LabelType, str] = {
    "introduction_clinicians": "Introduction of **all** clinicians involved in handover",
    "introduction_patients": (
        "Introduction of **all** clinicians involved in handover to patient/carer"
    ),
    "invitation": "Invitation for patient/carer to participate in handover",
    "id_check": "ID check of 3 patient identifiers (name, date of birth and ID number)",
    "sit_diagnosis": "Primary diagnosis; reason for admission",
    "sit_events": "Significant events or complications",
    "sit_status": (
        "Current status (awaiting tests/procedures, on interim orders/plan)"
    ),
    "bg_history": ("Relevant clinical and social history; comorbidities"),
    "bg_fall_risk": "Alerts - falls risk",
    "bg_pi_risk": "Alerts - pressure injury risk",
    "bg_allergies": "Alerts - allergies",
    "bg_acp": "Advanced care planning",
    "assess_obs": "Observations; Q-ADDS; recent escalations",
    "assess_pain": "Pain management",
    "assess_devices": "Devices; lines; vascular access",
    "assess_monitoring": "Critical monitoring; alarms",
    "assess_nutrition": "Nutrition; restrictions",
    "assess_fluid_balance": "Fluid balance; restrictions",
    "assess_infusions": "Infusions",
    "assess_medications": "Medication chart; flag high risk meds",
    "assess_pathology": "Pathology",
    "assess_mobility": "Mobility; aids",
    "assess_skin_integrity": "Skin integrity; interventions",
    "rec_discharge_plan": "Discharge plan",
    "rec_actions": "Critical actions required",
    "rec_plan": "Care plan/pathway actions to follow up",
    "rec_patient_goals": ("Asked patient/carer about goals and preferences"),
}


def build_label_desc():
    lines = [
        "Choose zero or more labels and return a Python list.",
        "",
        "Available labels:",
    ]
    for k, v in LABEL_DESCRIPTIONS.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)
