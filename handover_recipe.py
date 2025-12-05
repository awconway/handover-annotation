import prodigy
import spacy
from prodigy.components.stream import get_stream
from prodigy.components.preprocess import add_tokens


def remove_token_spans(examples):
    for eg in examples:
        eg.pop("tokens", None)  # remove the tokens key
    return examples


@prodigy.recipe("handover_spancat")
def handover_spancat():
    stream = get_stream("handover.jsonl")
    nlp = spacy.blank("en")  # blank spaCy pipeline for tokenization
    stream.apply(add_tokens, nlp=nlp, stream=stream)

    blocks = [
        {
            "view_id": "html",
            "html_template": "<div style='border-top: 5px solid #ddd;padding-top: 15px;'><h4 style='text-align: left;'>Task 1: Label the handover text by selecting a label and highlighting the relevant information.</h4> <p style='text-align: left;'>Note: This task involves categorising the text into the SBAR format <strong>and</strong> also labelling parts of the text where uncertain or vague pieces of information were communicated, which may signal a requirement for verification. More than one label can be applied to overlapping parts of the text. Review the help section for examples (question mark icon top left).</p></div>",
        },
        {
            "view_id": "spans_manual",
            "labels": [
                "SITUATION",
                "BACKGROUND",
                "ASSESSMENT",
                "RECOMMENDATION",
                "Vagueness",
                "Hedging",
                "Unknown fact",
                "Indefinite timing",
                "Source uncertainty",
                "Procedural uncertainty",
                "Responsibility uncertainty",
            ],
        },
        {
            "view_id": "html",
            "html_template": "<h4 style='text-align: left;border-top: 5px solid #ddd;padding: 15px;'>Task 2: Select all of the aspects that were addressed in the handover</h4>",
        },
        {
            "view_id": "choice",
            "text": None,
            "options": [
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
            ],
        },
        {
            "view_id": "html",
            "html_template": "<div style='border-top: 5px solid #ddd;padding: 15px;'><h4 style='text-align: left;'>Task 3: From the perspective of the person receiving the handover, provide 3 additional questions you would consider asking before completing the handover interaction.</h4></div>",
        },
        {
            "view_id": "text_input",
            "field_rows": 2,
            "field_label": "Question one",
            "field_placeholder": "Type example question here",
            "field_id": "question_1",
            "field_autofocus": False,
        },
        {
            "view_id": "text_input",
            "field_rows": 2,
            "field_label": "Rationale for question one",
            "field_placeholder": "Type rationale for question here",
            "field_id": "rationale_1",
            "field_autofocus": False,
        },
        {
            "view_id": "html",
            "html_template": "<div style='border-top: 1px solid #ddd;'></div>",
        },
        {
            "view_id": "text_input",
            "field_rows": 2,
            "field_label": "Question two",
            "field_placeholder": "Type example question here",
            "field_id": "question_2",
            "field_autofocus": False,
        },
        {
            "view_id": "text_input",
            "field_rows": 2,
            "field_label": "Rationale for question two",
            "field_placeholder": "Type rationale for question here",
            "field_id": "rationale_2",
            "field_autofocus": False,
        },
        {
            "view_id": "html",
            "html_template": "<div style='border-top: 1px solid #ddd;'></div>",
        },
        {
            "view_id": "text_input",
            "field_rows": 2,
            "field_label": "Question three",
            "field_placeholder": "Type example question here",
            "field_id": "question_3",
            "field_autofocus": False,
        },
        {
            "view_id": "text_input",
            "field_rows": 2,
            "field_label": "Rationale for question three",
            "field_placeholder": "Type rationale for question here",
            "field_id": "rationale_3",
            "field_autofocus": False,
        },
    ]

    return {
        "view_id": "blocks",
        "dataset": "handover_db_dev",
        "stream": stream,
        # "before_db": remove_token_spans,
        "config": {
            "blocks": blocks,
            "keymap": {"undo": ["z", "backspace"]},
            "annotations_per_task": 2,
            "choice_style": "multiple",
            "buttons": ["accept"],
            "show_stats": False,
            "theme": "spacy",
            "instructions": "instructions.html",
        },
    }
