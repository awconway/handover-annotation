from typing import Literal, TypedDict

import dspy

LabelType = Literal["UNCERTAIN"]


class LabelQuote(TypedDict):
    label: LabelType
    quote: str


class LabelHandover(dspy.Signature):
    """
    To extract quotes from the text of a clinical handover transcript that contain uncertain information.

    Uncertainty includes:
    - Hedge / Probability Language (e.g., "I think ENT reviewed him.")
    - Vague / Qualitative Expression (e.g., "He looks fine now.")
    - Unknown fact / Explicit Lack of Knowledge (e.g., "Not sure if consent’s been signed.")
    - Indefinite Timing (e.g., "Later today.")
    - Source Uncertainty (e.g., "ENT said he’s on the list.")
    - Procedural Uncertainty (e.g., "We’ll see how he goes.")
    - Responsibility Uncertainty (e.g., "Bloods to be checked later.")

    Instructions
    - Extract each uncertain piece of information as a separate quote.
    - Label every extracted quote as UNCERTAIN.
    - **Do not combine parts from different sections of the text into a single quote.**
    - Extracted quotes must appear **exactly as written in the original transcript**, preserving spelling, capitalization, punctuation, and phrasing. Do not paraphrase. Do not summarize. Do not correct incorrectly spelled words. Do not correct errors. Do not rephrase.
    - If text is not uncertain, do not extract it.
    """

    text: str = dspy.InputField()
    pred_spans: list[LabelQuote] = dspy.OutputField()


def build_predictor():
    return dspy.Predict(LabelHandover)
