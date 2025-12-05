import dspy

from checklist_task.labels import LabelType, build_label_desc


class LabelHandover(dspy.Signature):
    """
    Determine which categories have been addressed in the transcript of a clinical handover.

    | Key                     | Description                                                         |
    |-------------------------|---------------------------------------------------------------------|
    | introduction_clinicians | Introduction of **all** clinicians involved in handover                    |
    | introduction_patients   | Introduction of **all** clinicians involved in handover to patient/carer   |
    | invitation              | Invitation for patient/carer to participate in handover            |
    | id_check                | ID check of 3 patient identifiers (name, date of birth, and ID number)                                  |
    | sit_diagnosis           | Primary diagnosis; reason for admission                          |
    | sit_events              | Significant events or complications                                |
    | sit_status              | Current status (awaiting tests/procedures, on interim orders/plan) |
    | bg_history              | Relevant clinical and social history; comorbidities              |
    | bg_fall_risk            | Alerts - falls risk                                                |
    | bg_pi_risk              | Alerts - pressure injury risk                                      |
    | bg_allergies            | Alerts - allergies                                                 |
    | bg_acp                  | Advanced care planning                                             |
    | assess_obs              | Observations; Q-ADDS; recent escalations                       |
    | assess_pain             | Pain management                                                    |
    | assess_devices          | Devices; lines; vascular access                                |
    | assess_monitoring       | Critical monitoring; alarms                                      |
    | assess_nutrition        | Nutrition; restrictions                                          |
    | assess_fluid_balance    | Fluid balance; restrictions                                      |
    | assess_infusions        | Infusions                                                          |
    | assess_medications      | Medication chart; flag high risk meds                            |
    | assess_pathology        | Pathology                                                          |
    | assess_mobility         | Mobility; aids                                                   |
    | assess_skin_integrity   | Skin integrity; interventions                                    |
    | rec_discharge_plan      | Discharge plan                                                     |
    | rec_actions             | Critical actions required                                          |
    | rec_plan                | Care plan/pathway actions to follow up                             |
    | rec_patient_goals       | Asked patient/carer about goals and preferences                    |
    """

    text: str = dspy.InputField()
    labels: list[LabelType] = dspy.OutputField(desc=build_label_desc())


def build_predictor():
    return dspy.Predict(LabelHandover)
