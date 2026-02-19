import os
from typing import Literal

from typing_extensions import TypedDict

import dspy

LabelType = Literal[
    "Vagueness",
    "Hedging",
    "Unknown fact",
    "Indefinite timing",
    "Source uncertainty",
    "Procedural uncertainty",
    "Responsibility uncertainty",
]


class LabelQuote(TypedDict):
    label: LabelType
    quote: str


class LabelHandover(dspy.Signature):
    """
    To extract quotes from the text of a clinical handover transcript that contains uncertain information.

    | Category | Definition / When to Use | Example From Handover Speech |
    |---------|---------------------------|-------------------------------|
    | **Hedge / Probability Language** | The speaker indicates partial confidence or doubt about information. | “I think ENT reviewed him.” “He should be going to theatre soon.” |
    | **Vague / Qualitative Expression** | Information is described using imprecise or subjective language. | “He looks fine now.” “Seems okay.” |
    | **Unknown fact / Explicit Lack of Knowledge** | The speaker openly states missing knowledge or incomplete data. | “Not sure if consent’s been signed.” “I don’t know his allergies.” |
    | **Indefinite Timing** | Timing or schedule for an event is vague or lacks precision. | “Later today.” “After the round.” |
    | **Source Uncertainty** | Information relies on a second-hand or unverifiable source. | “ENT said he’s on the list.” “Night nurse told me.” |
    | **Procedural Uncertainty** | Unclear next step in process or lack of explicitly stated plan. | “We’ll see how he goes.” “You might want to check his IV.” |
    | **Responsibility Uncertainty** | A required task or follow-up is mentioned, but **it’s unclear who is responsible** for  performing it. | “Bloods to be checked later.”“Needs review this afternoon.” |

    Instructions
    - Extract each uncertain piece of information as a separate quote.
    - The same quote can be labelled into more than one of the uncertainty categories.
    - **Do not combine parts from different sections of the text into a single quote.**
    - Extracted quotes must appear **exactly as written in the original transcript**, preserving spelling, capitalization, punctuation, and phrasing. Do not paraphrase. Do not summarize. Do not correct incorrectly spelled words. Do not correct errors. Do not rephrase.
    - If text is not uncertain, do not extract it.
    """

    text: str = dspy.InputField()
    pred_spans: list[LabelQuote] = dspy.OutputField()


class LabelHandoverGepaMedgemma(dspy.Signature):
    """
    Extract uncertain statements from a clinical handover transcript.

    Allowed labels (must match exactly):
    - Vagueness
    - Hedging
    - Unknown fact
    - Indefinite timing
    - Source uncertainty
    - Procedural uncertainty
    - Responsibility uncertainty

    Instructions
    - Return `pred_spans` as a list of JSON objects with exactly two fields: `label` and `quote`.
    - `quote` must be an exact substring from the source text (no paraphrasing, no corrections).
    - Use one item per uncertain statement; do not merge different text segments.
    - If one quote matches multiple uncertainty categories, repeat the quote with different labels.
    - If there is no uncertainty, return an empty list.
    """

    text: str = dspy.InputField()
    pred_spans: list[LabelQuote] = dspy.OutputField()


_SIGNATURE_REGISTRY = {
    "default": LabelHandover,
    "gepa_medgemma": LabelHandoverGepaMedgemma,
    "gepa": LabelHandoverGepaMedgemma,
}


def build_predictor(signature_mode: str | None = None):
    mode = (signature_mode or os.getenv("UNCERTAIN_SIGNATURE_MODE", "default")).strip().lower()
    if mode not in _SIGNATURE_REGISTRY:
        allowed = ", ".join(sorted(_SIGNATURE_REGISTRY))
        raise ValueError(
            f"Unknown uncertainty signature mode: {mode!r}. "
            f"Expected one of: {allowed}."
        )
    return dspy.Predict(_SIGNATURE_REGISTRY[mode])
