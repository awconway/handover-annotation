---
title: "Towards AI-assisted clinical handover: Evaluating large language models for structured information extraction"

authors:
  - name: Aaron Conway
    affiliations:
      - ref: qut
      - ref: tpch
    corresponding: true
    email: aaron.conway@utoronto.ca
    orcid: 0000-0002-9583-8636


affiliations:
  - id: qut
    name: Centre for Healthcare Transformation, Queensland University of Technology, Brisbane, Australia
  - id: tpch
    name: The Prince Charles Hospital, Brisbane, Australia


filters:
  - authors-block
  - checklist-table-layout

format:
  docx:
    reference-doc: custom-reference-doc.docx
    fig-width: 8
bibliography: references.bib
jupyter: python3
execute:
  echo: false
  warning: false
  error: false
  keep-md: true
---

{{< pagebreak >}}


## Abstract {#sec-abstract}


### Introduction

Artificial intelligence could be used to support clinical handover by producing summaries in structured formats, verifying that key information has been communicated and signaling information that may require verification. We evaluated the performance of several different large language models and prompt optimisation strategies for structured information extraction across tasks that could be integrated into AI-assisted clinical handover.

### Methods

Two registered nurses independently annotated a dataset of 203 synthetic handover transcripts to produce consensus labels for each information extraction task. Tasks included: 1) labelling spans of text into SBAR (Situation, Background, Assessment, Recommendation) categories; 2)  judging if specific pieces of information were communicated as a form of content detection; and 3) labelling spans of text that communicated information using uncertain terms, including a sub-task for identifying unknown facts. Baseline and Genetic-Pareto (GEPA) optimised prompts were compared for GPT-5.2, GPT-5-nano, and MedGemma 27B large langage models. Additionally, the LangExtract framework was evaluated for span-extraction tasks. Content detection was scored using micro-, macro-, and support-weighted precision, recall, and F1; span tasks were scored using same-label quote matching, matched-span precision/recall/F1, and mean intersection-over-union for boundary agreement.

### Results

In matched GPT-5.2 comparisons, prompt optimisation increased micro-F1 by \+0\.08 for content detection, \+0\.24 for SBAR span extraction, \+0\.06 for uncertainty-span extraction, and \+0\.08 for unknown-fact extraction. The GPT-5.2 optimised model achieved micro-F1 0\.85 and support-weighted F1 0\.85 for content detetection. For SBAR span extraction, GPT-5.2 with GEPA prompt optimisation achieved micro-F1 0\.76 and macro-F1 0\.75, compared with baseline macro-F1 0\.49. Broad uncertainty-span extraction remained comparatively weak (micro-F1 0\.41), whereas explicit unknown-fact extraction was stronger (micro-F1 0\.84). Prompt optimisation outperformed the LangExtract approach across each task.

### Conclusion

Prompt-optimisation enhanced the performance of large language models for tasks that can be used in real-time clinical handover support. Broad uncertainty detection was not sufficiently reliable for deployment. For most tasks, the smaller GPT-5-nano model performed competitively with the larger GPT-5.2, suggesting that cost-effective models may be viable for certain applications. Future work should explore integration of these capabilities into clinical workflows and evaluate their impact on communication quality and patient outcomes.

## Introduction {#sec-introduction}

Clinical handover is a safety-critical transition in which responsibility and accountability for patient care are transferred between clinicians. Failures in communication that occur at handover through inaccurate, incomplete, or misinterpreted information transfer create opportunities for delays, duplicated work, and preventable harm [@ong2011handoff_failures; @manias2016handover_perspectives]. In recognition of this risk, international and national bodies have prioritised structured handover practices and communication redesign as patient-safety interventions, particularly at transitions of care where information loss is more likely [@manser2011effective_handover; @bukoh2020structured_handover_review].

Structured handover processes aim to standardise both the minimum content and the format of exchange so that critical information is predictably conveyed, acted upon, and auditable. In Australia, the Communicating for Safety Standard emphasises structured clinical handover to reduce communication errors and improve patient safety, explicitly noting the heightened risk at shift changes, transfers, and discharge [@redley2018handover_audit_tool; @manias2016handover_perspectives]. One widely adopted format is SBAR (Situation, Background, Assessment, Recommendation), framed as a shared mental model to support concise, organised clinician-to-clinician communication [@haig2006sbar_shared_mental_model]. However, despite widespread uptake, the evidence base for SBAR’s direct impact on patient outcomes varies by context, and systematic review findings have been described as moderate with ongoing calls for higher-quality evaluation [@muller2018sbar_systematic_review].

Even where structured tools are mandated or encouraged, handover content remains influenced by time pressure, local culture, clinician experience, and the pragmatic realities of ward work [@riesenberg2010nursing_handoffs; @watson2014time_handover; @manias2016handover_perspectives]. This variability is amplified in bedside nursing handover where patient/family involvement is increasingly emphasised, yet research highlights tensions between standardisation (predictability) and tailoring (patient-centredness), along with barriers related to confidentiality and clinician concerns [@tobiano2018patient_participation_bedside_handover].

Recent advances in large language models (LLMs) create an opportunity to facilitate real-time support of clinical communication and decision-making during handover. The aim of this study was to evaluate the accuracy of LLM extraction of structured information from clinical handover transcripts across tasks aligned with real-time clinical handover support.

## Methods {#sec-methods}

Tasks related to clinical handover communication that were considered as potentially augmentable with AI assistance were identified by the researchers using a co-design process with clinicians and consumers, which will be reported separately. The tasks were:

- Extracting spans of text from transcripts that aligned with the SBAR (Situation, Background, Assessment, Recommendation) framework for structuring clinical handover. In this study, SBAR span extraction was operationalised as a sequence-labeling task to identify contiguous spans of text in handover transcripts that corresponded to each SBAR category.
- Identifying if key elements were communicated in handover transcripts as a content detection task. In this study, this task was operationalised as a checklist of items that are recommended to be addressed during nursing clinical handover, which were developed as part of quality-improvement processes at the researchers' institution. The checklist items are summarised in @tbl-checklist-items.
- Identifying spans of text in transcripts that were communicated using uncertain terms. In this study, this task was operationalised as span annotation of utterances during handover that conveyed incomplete knowledge, vague or hedged wording, imprecise timing, second-hand sourcing, unclear procedures, or unclear responsibility for follow-up actions. These forms of uncertainty were treated as potentially clinically important because they may indicate information that requires clarification or verification by the receiving clinician. The uncertainty categories and example guidance provided to annotators are summarised in @tbl-uncertainty-items. 

### Data sources

We used the publicly available NICTA Synthetic Nursing Handover Dataset, which contains realistic recordings of clinical handovers delivered by a registered nurse based on patient profiles with cardiovascular, neurological, renal, and respiratory conditions [@suominen2015benchmarking]. In this dataset, handover monologues were generated from comprehensive patient profiles that included information such as the patient’s name, age, admission history, inpatient duration, and the familiarity between the nurses giving and receiving the handover. The nurse was instructed to simulate a bedside shift-to-shift handover within a medical ward setting [@suominen2015benchmarking].

For our study, we used 100 handover samples from the training partition of the NICTA dataset. First, audio recordings from the NICTA dataset were transcribed using the OpenAI Whisper speech-to-text model. Second, three videos depicting conversational nursing shift-to-shift handovers were transcribed in a similar manner to provide examples of interactive handover dialogue. These videos were developed at the authors’ institution for educational purposes to demonstrate best practices for clinical handovers that are used in the undergraduate nursing program. Transcripts from the educational videos were then used as few-shot examples within the DSPy framework[@khattab2023dspy]] using the `BootstrapFewShot` optimizer to guide the transformation of the 100 NICTA monologue transcripts into two-sided conversational handovers so that the dataset would better reflect real-world contemporary clinical handover interactions. To broaden the range of clinical contexts represented in the dataset, we further synthesised 103 additional handover examples using the GPT-5 model interactively in a chat interface. These scenarios included inter- and intra-hospital transfers, post-procedural handovers, emergency department transitions, and handovers that involved patients with complex mental health care needs. The final dataset comprised 203 synthetic handover transcripts used for subsequent annotation and model development. It is available at [URL to dataset repository]. 

### Annotation

Two annotators, who are experienced clinically active Registered Nurses, independently annotated transcripts using a structured rubric. Labelled annotations that met consensus between reviewers were used as the reference labels for downstream model development and evaluation.

Annotation was performed using Prodigy annotation software using a custom interface that presented each transcript in three components [@montani2018prodigy]. First, annotators highlighted relevant spans of text and assigned labels corresponding to the SBAR framework together with markers of communicative uncertainty, including vagueness, hedging, unknown facts, indefinite timing, source uncertainty, procedural uncertainty, and uncertainty regarding responsibility. Overlapping span labels were permitted where a passage served more than one communicative function. Second, annotators completed a multiple-response checklist indicating whether predefined handover elements were present in the transcript. Third, annotators recorded up to three additional questions that a receiving clinician might reasonably ask before concluding the handover, together with a brief rationale for each question, in order to capture perceived information gaps.

Both annotators reviewed all transcripts independently within the same annotation environment and were supported by written guidance and examples to promote consistent interpretation of the coding framework. For creation of the reference standard, a consensus dataset was derived by retaining only those span annotations and checklist items for which both annotators agreed. In practical terms, this meant that only text segments assigned the same label by both reviewers, and only checklist items selected by both reviewers, were carried forward for downstream model development and evaluation.

### Large language model prompt optimisation methods

Annotated handover transcripts were first partitioned deterministically into optimisation (75%) and evaluation (25%) subsets using a fixed random split. Prompt optimisation was performed within DSPy [@khattab2023dspy], which is a Python framework that can be used for prompt optimisation using feedback from model outputs to improve task performance. For each task, we defined a task-specific DSPy signature that specified the transcript as input and a constrained structured output. Checklist prediction was formulated as multilabel classification, whereas SBAR and uncertainty-related tasks were formulated as span extraction, requiring the model to return verbatim text segments from the transcript together with the appropriate label. The uncertainty-related tasks were further subdivided into a broad uncertainty-span extraction task (which included all uncertainty categories) and a more specific unknown-fact extraction task (which included only spans labelled as unknown facts).

Across the baseline DSPy evaluations, GEPA-based DSPy optimisation experiments, and LangExtract experiments, we evaluated three underlying language models: OpenAI GPT-5-nano [@singh2025openaigpt5card], which served as a smaller lower-cost comparison model; OpenAI GPT-5.2 [@singh2025openaigpt5card], which served as the highest-capacity proprietary model evaluated in this study; and MedGemma 27B, a 27-billion-parameter open-weight model developed by Google for medical tasks [@sellergren2025medgemmatechnicalreport]. For DSPy baseline and GEPA runs, the same task model was used before and after compilation so that differences reflected prompt optimisation rather than a change in the underlying model.

Baseline performance was obtained by evaluating the initial DSPy signatures. Optimised performance was obtained by compiling the same predictors on the optimisation partition. For checklist prediction, optimisation targeted example-level multilabel F1 against the labels. A prompt optimisation method called GEPA was used to optimise the prompts [@agrawal2025gepa]. GEPA is a reflection-based evolutionary prompt optimiser within DSPy [@agrawal2025gepa]. This approach involves iteratively proposing revised task instructions and retains variants that improve task-specific performance. The optimisation was conducted over 576 scoring calls of the full program, which is equivalent to about 3 complete scoring passes over the combined training and validation examples used during prompt search. GEPA was supplied with both a scalar score and natural-language error feedback. For checklist prediction, feedback identified which labels had been correctly included, incorrectly added, or omitted. For span-extraction tasks, feedback described the reference spans, predicted spans, matched and unmatched spans, and the extent of span-boundary overlap. This feedback was then used by a separate reflection model (GPT-5.2) to generate revised instructions for the predictor. GPT-5.2 was selected as the reflection model because it was the strongest performing model available within our evaluation framework at the time. For SBAR and other span-extraction tasks, GEPA optimisation was driven by a metric based on intersection-over-union. In practical terms, intersection-over-union measured how much the predicted text span overlapped the annotated text span relative to the total text covered by either span. Higher values of intersection-over-union indicate closer agreement on the start and end boundaries of the labelled span. To calculate intersection-over-union, predicted quotes were mapped back to their location in the transcript and matched to reference spans of the same label according to overlap, with higher-overlap matches receiving greater reward during prompt search. In this way, prompt search favoured not only correct matching but also closer boundary alignment. When neither reference nor predicted spans were present for a transcript, the transcript-level optimisation score was set to 1.0. For reporting, we then separated detection performance from boundary quality by presenting matched-span precision, recall, and F1 together with mean IoU for matched spans. The final compiled programs were then fixed and evaluated on the held-out test partition.

In addition to DSPy prompt optimisations, we conducted separate evaluations using the LangExtract framework, as a prompt-based few-shot extraction approach for the SBAR, uncertainty-span, and unknown-fact span tasks [@goel_2026_langextract]. Within this framework, extraction experiments were run with GPT-5.2 and MedGemma 27B; GPT-5-nano was evaluated within the DSPy baseline and GEPA workflows rather than the LangExtract experiments. These experiments used task-specific prompt descriptions together with annotated in-context examples derived from the reference data. We used 10 annotated examples from the training partition as few-shot exemplars, and inference was then performed on the full held-out test partition for each task. Few-shot examples were validated to ensure alignment between the quoted extraction and the source transcript before inference. LangExtract outputs were normalised to the same label-and-verbatim-quote representation used for the DSPy span tasks and were scored with the same matching and overlap-based evaluation procedure, allowing direct comparison across methods. For the binary unknown-fact experiments, the few-shot example set was additionally constrained to include positive examples of the target label.

### Data analysis

Performance was measured with standard classification metrics for the checklist content detection task. At the level of individual labels, we calculated counts of true positives, false positives, false negatives, and true negatives, together with precision, recall, and F1. Aggregate performance was summarised using micro-averaged (pooled), macro-averaged (unweighted mean), and support-weighted precision, recall, and F1 across labels.

For span-extraction tasks, model outputs were represented as label-and-quote pairs, and each predicted quote was mapped back to a character span in the source transcript using exact string matching, with approximate matching used only when an exact match was unavailable. Predicted spans were then matched one-to-one with reference spans of the same label according to highest overlap. Under this greedy matching procedure, a reported match was defined as a same-label pair with non-zero overlap. Precision, recall, and F1 were then calculated from these matched gold and predicted spans as binary detection measures, so that these statistics reflected the model's ability to identify the correct labelled spans. Span-boundary agreement was reported separately using the mean intersection over union (IoU) across matched pairs. We additionally calculated per-label descriptive metrics including the number of reference spans, the number of predicted spans, matched-span precision, recall, F1, and mean IoU.


## Results {#sec-results}

### Key findings {#sec-results-summary}

Within-model comparisons showed consistent gains with DSPy/GEPA over matched baselines wherever both were available. For GPT-5.2, DSPy/GEPA improved micro-F1 by \+0\.08 for checklist prediction, \+0\.24 for SBAR span extraction, \+0\.06 for uncertainty span extraction, and \+0\.08 for unknown-fact extraction. GPT-5-nano also improved over baseline for checklist prediction (\+0\.09), SBAR span extraction (\+0\.20), and unknown-fact extraction (\+0\.71). MedGemma 27B improved in each matched DSPy/GEPA comparison: checklist prediction (\+0\.11), SBAR span extraction (\+0\.16), uncertainty span extraction (\+0\.06), and unknown-fact extraction (\+0\.37). Across span tasks, LangExtract generally performed below the corresponding best DSPy/GEPA configuration.

### Comparative performance across tasks and prompting approaches {#sec-results-overview}

@fig-results-overview summarizes matched within-model comparisons across tasks for the three models evaluated in the main study. 


### SBAR span extraction {#sec-sbar-optimized}

Among SBAR configurations, the highest overall score was achieved by DSPy/GEPA-optimised GPT-5.2 (micro-F1 0\.76), followed by DSPy/GEPA-optimised GPT-5-nano (micro-F1 0\.69) and LangExtract GPT-5.2 (micro-F1 0\.59). @tbl-sbar-optimized provides label-level results for the best-performing GPT-5.2 DSPy/GEPA SBAR configuration, compared with its corresponding GPT-5.2 baseline run.


Within this GPT-5.2 SBAR comparison, macro-precision improved from 0\.41 to 0\.78, macro-recall from 0\.69 to 0\.73, and macro-F1 from 0\.49 to 0\.75. The largest per-label improvements in F1 were observed for BACKGROUND (\+0\.40), SITUATION (\+0\.25), and ASSESSMENT (\+0\.23), with a smaller improvement for RECOMMENDATION (\+0\.15). Span-boundary agreement among matched predictions remained high, with mean IoU values between 0\.66 and 0\.82 across SBAR labels.

### Checklist task {#sec-checklist}

For checklist prediction, the best overall result was achieved by DSPy/GEPA-optimised GPT-5.2 (micro-F1 0\.85, macro-F1 0\.76, support-weighted F1 0\.85). This exceeded the matched GPT-5.2 baseline by \+0\.08 on micro-F1 and \+0\.07 on macro-F1. DSPy/GEPA-optimised GPT-5-nano also performed competitively (micro-F1 0\.81), while MedGemma 27B reached micro-F1 0\.76). @tbl-checklist-grouped presents grouped per-label performance for the best-performing GPT-5.2 checklist model. The largest improvements over baseline were observed for patient involvement items and several recommendation-related labels, while 2 low-support labels still had F1 equal to zero.


## Discussion {#sec-discussion}

This study identified that prompt optimisation with GEPA outperformed baseline evaluations across all of the comparisons, with higher aggregate performance for checklist prediction than for SBAR span extraction. This pattern is expected because using generative AI to perform clinical natural language processing tasks is sensitive to framing, label definitions, and example selection. GEPA was explicitly supplied with task-specific scores and error feedback that could align instructions more closely with the annotation scheme [@sivarajkumar2024prompting_strategies_clinical_nlp; @agrawal2025gepa]. Optimisation was particularly useful for improving targeted span extraction across the SBAR task.

The difference in accuracy that we identified between checklist prediction and SBAR span extraction provides an important insight to consider for designing AI applications to support clinical handover. Checklist prediction is a simpler structured-output task where each item is a bounded present/absent judgement. By contrast, SBAR extraction requires the model to locate clinically relevant text, assign a communicative category, and reproduce appropriate span boundaries. Structured handover tools aim to make minimum content predictable and reduce omissions, while still requiring adaptation to local clinical workflow [@haig2006sbar_shared_mental_model; @riesenberg2010nursing_handoffs; @bukoh2020structured_handover_review]. As such, simpler structured outputs may therefore be preferable for an AI handover tool when the intended function is to prompt review of missing elements or support audit and quality monitoring. However, this type of structured data may be overly rigid structure, reducing narrative expressivity and the overall fit with clinical work [@rosenbloom2011structured_flexible_documentation]. In addition, although checklist prediction achieved strong pooled performance, the model still produced both false positives and false negatives. A false positive could create false reassurance that a safety-critical element was communicated during handover, whereas a false negative could generate unnecessary clarification work or distract from higher-priority issues. A safer near-term design could therefore present AI outputs as draft, source-linked prompts for clinician review, with clear distinctions between detected in transcript, not detected, and requires verification. Evidence indicated that effective clinical decision support is most useful when integrated into a workflow as actionable and readily available support at the time and place of decision-making, while also recognising the risk of automation bias when clinicians over-rely on system outputs [@kawamoto2005clinical_decision_support; @goddard2012automation_bias; @challen2019ai_bias_safety].

In the same way, SBAR span extraction may be most useful as a traceability mechanism rather than as a fully automated handover summary. The optimised model achieved useful SBAR performance and high span-boundary overlap among matched spans in our study. For implementation, this suggests that extracted spans could be displayed with a link to their source transcript context and should support clinician review of what was said, rather than automatically transforming the handover into a definitive structured note. This preserves the benefits of structured communication while avoiding over-compression of clinically meaningful narrative context [@haig2006sbar_shared_mental_model; @rosenbloom2011structured_flexible_documentation; @cohen2010handoffs_literature_review].

The appropriate balance between recall and precision may differ by use case when considering implementation of the tasks we evaluated in this study into an AI tool to support clinical handover. For real-time clinician support, higher recall may be acceptable if prompts are phrased as optional checks and do not imply that an item is definitely missing. For retrospective audit or compliance monitoring, precision becomes more important because false positives could overestimate handover quality and obscure residual safety risks. The present checklist results, with strong recall but non-trivial false positives in several categories, therefore support cautious separation of two implementation pathways: real-time clinician-facing gap prompts, and separately validated audit/reporting workflows [@redley2018handover_audit_tool; @manser2011effective_handover; @kawamoto2005clinical_decision_support]

Gold standard handovers include active verification and shared situational awareness rather than passive transfer of uncertain information [@patterson2004handoff_strategies; @cohen2010handoffs_literature_review; @leonard2004human_factor; @starmer2014handoff_program]. Evaluation results of the broad uncertainty-span performance indicates that even a GEPA optimised prompt with a state of the art propietary large language model is not yet reliable for detecting the full range of ambiguous, hedged, or context-dependent communication. By contrast, the comparatively stronger unknown-fact extraction results suggest a potentailly useful role for identifying explicit information gaps that can be converted into clarification prompts for the receiving clinician. As foundation models continue to improve, accuracy on this task may also improve, although reliable use in safety-critical handover settings will still require ongoing empirical validation.

Relatedly, it is highly likely that further gains could be realised after implementation independent of general improvements in underlying large language model capabilty. If clinicians review AI-generated handover outputs and corrections are retained as labelled examples, subsequent optimisation cycles using the same GEPA prompt optimisation pipeline we used in this study would plausibly improve performance over time. Prior clinical natural language processing active-learning studies have demonstrated that selectively labelled examples can improve data efficiency for both clinical text classification and clinical named-entity recognition [@figueroa2012active_learning_clinical_text; @chen2015active_learning_ner_clinical_text]. In practice, any such learning cycle should be treated as a controlled quality-improvement process, with monitoring and re-evaluation before revised prompts or models are released into clinical use [@feng2022clinical_ai_quality_improvement].

Finally, it should be noted that this study measured extraction and classification performance against annotated labels, but did not evaluate whether AI outputs changed clinician questioning, closed-loop communication, task completion, escalation, near-miss detection, or patient outcomes. Technical metrics are necessary but insufficient for determining whether a clinical AI tool improves safety in practice [@kelly2019clinical_ai_impact; @challen2019ai_bias_safety; @starmer2014handoff_program] Even accurate prompts may create new risks if clinicians over-trust them, ignore non-highlighted issues, experience alert fatigue, or redirect attention away from direct patient/carer engagement. Human factors testing should therefore assess reliance, trust calibration, interruption burden, and the usability of source-linked evidence before deployment [@goddard2012automation_bias; @leonard2004human_factor; @sittig2010sociotechnical_model].

## Limitations {#sec-limitations}


The synthetic conversational handovers used in this study may not fully represent those performed in actual clinical practice where interruptions, time pressure, environmental noise, non-verbal cues and local team dynamics can affect what is communicated and how it is interpreted. External validation on real handover audio/transcripts from intended deployment settings is therefore required [@cohen2010handoffs_literature_review; @patterson2004handoff_strategies; @sittig2010sociotechnical_model]. It sould be noted that several checklist items had low prevalence in the evaluation partition, including the three-patient-identifier verification, advance care planning, critical monitoring/alarms, and critical actions required. Reliable implementation decisions cannot be made from aggregate micro-F1 when rare but high-consequence labels are underrepresented. Future evaluation should oversample or otherwise specifically test these items [@challen2019ai_bias_safety; @riesenberg2010nursing_handoffs]. The reference standard retained only annotations agreed by both reviewers. This creates a conservative and reproducible benchmark, but may exclude ambiguous or contested communication—precisely the type of handover content that can be clinically important. As a result, model performance may be overestimated for clear-cut cases and less informative for the ambiguous, implicit, or disputed information that often drives clarification needs during real handover [@cohen2010handoffs_literature_review; @manser2011effective_handover]. Our evaluation focused on downstream extraction from transcripts and did not separately quantify transcription accuracy, speaker attribution, or the effect of noisy audio. In a real-time handover system, errors introduced before the LLM stage could alter checklist detection, SBAR span extraction, and uncertainty identification. End-to-end evaluation should therefore include audio capture, transcription, diarisation, and transcript-to-output performance under realistic clinical conditions [@sittig2010sociotechnical_model; @ong2011handoff_failures]. We used a checklist that was developed at a local institution based on local priorities for communication during clinical nursing handover. Other wards, specialties, transfer types, or jurisdictions may prioritise different minimum datasets or use different terminology. Implementation would therefore require local mapping of labels to existing handover policy, audit tools, escalation pathways, and documentation workflows, followed by local validation rather than direct transfer of the reported performance estimates [@redley2018handover_audit_tool; @riesenberg2010nursing_handoffs; @sittig2010sociotechnical_model].

## Conclusion {#sec-conclusion}

In this evaluation of AI-assisted clinical handover tasks, DSPy/GEPA prompt optimisation consistently improved matched model performance and was most promising for bounded checklist prediction and SBAR span extraction. The results support an implementation pathway in which AI outputs are presented as clinician-reviewed, source-linked draft structures or gap prompts. Broad uncertainty detection remained insufficiently reliable, while explicit unknown-fact extraction suggests a narrower role for prompting clarification when missing knowledge is directly expressed. Before clinical deployment, these approaches require external validation on real handover audio and transcripts, end-to-end testing of transcription and diarisation effects, local calibration of checklist definitions, and human-factors evaluation of trust, reliance, workflow burden, and downstream safety outcomes.

{{< pagebreak >}}

# References

::: {#refs}
:::

{{< pagebreak >}}

::: {#fig-results-overview}

<svg xmlns="http://www.w3.org/2000/svg" width="980" height="576" viewBox="0 0 972 571.3" role="img" aria-labelledby="comparison-title comparison-desc" style="max-width:100%;height:auto;"><title id="comparison-title">Within-model comparison of baseline, DSPy/GEPA, and LangExtract micro-F1 performance</title><desc id="comparison-desc">Four-panel grouped horizontal bar chart across checklist, SBAR, uncertainty, and unknown-fact tasks.</desc><style>.plot-title{font:700 14px system-ui, sans-serif; fill:#192038;}.panel-title{font:700 12px system-ui, sans-serif; fill:#192038;}.axis-label{font:10px system-ui, sans-serif; fill:#4d5770;}.tick-label{font:10px system-ui, sans-serif; fill:#4d5770;}.model-label{font:11px system-ui, sans-serif; fill:#192038;}.legend-label{font:11px system-ui, sans-serif; fill:#192038;}.value-label{font:10px system-ui, sans-serif; fill:#2b3345;}</style><text x="22" y="38" class="plot-title">Within-model micro-F1 comparisons across tasks</text><rect x="22" y="50" width="16" height="10" fill="#8b95a7" rx="2" ry="2" /><text x="46" y="59" class="legend-label">Baseline</text><rect x="154" y="50" width="16" height="10" fill="#1f6feb" rx="2" ry="2" /><text x="178" y="59" class="legend-label">DSPy/GEPA</text><rect x="286" y="50" width="16" height="10" fill="#d97706" rx="2" ry="2" /><text x="310" y="59" class="legend-label">LangExtract</text><rect x="22" y="82" width="450" height="220" rx="10" ry="10" fill="#ffffff" stroke="#d8dcef" stroke-width="1.2" /><text x="38" y="102" class="panel-title">Checklist item prediction</text><line x1="172.0" y1="118.0" x2="172.0" y2="278.0" stroke="#d8dcef" stroke-width="1" /><text x="172.0" y="294.0" text-anchor="middle" class="tick-label">0.00</text><line x1="291.0" y1="118.0" x2="291.0" y2="278.0" stroke="#d8dcef" stroke-width="1" /><text x="291.0" y="294.0" text-anchor="middle" class="tick-label">0.50</text><line x1="410.0" y1="118.0" x2="410.0" y2="278.0" stroke="#d8dcef" stroke-width="1" /><text x="410.0" y="294.0" text-anchor="middle" class="tick-label">1.00</text><text x="291.0" y="284.0" text-anchor="middle" class="axis-label">Micro-F1</text><text x="160.0" y="144.0" text-anchor="end" class="model-label">GPT-5.2</text><rect x="172.0" y="126.0" width="184.2" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="362.2" y="135.0" class="value-label">0.77</text><rect x="172.0" y="140.0" width="203.4" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="381.4" y="149.0" class="value-label">0.85</text><text x="160.0" y="188.0" text-anchor="end" class="model-label">GPT-5-nano</text><rect x="172.0" y="170.0" width="171.5" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="349.5" y="179.0" class="value-label">0.72</text><rect x="172.0" y="184.0" width="193.5" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="371.5" y="193.0" class="value-label">0.81</text><text x="160.0" y="232.0" text-anchor="end" class="model-label">MedGemma 27B</text><rect x="172.0" y="214.0" width="154.5" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="332.5" y="223.0" class="value-label">0.65</text><rect x="172.0" y="228.0" width="181.8" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="359.8" y="237.0" class="value-label">0.76</text><rect x="500" y="82" width="450" height="220" rx="10" ry="10" fill="#ffffff" stroke="#d8dcef" stroke-width="1.2" /><text x="516" y="102" class="panel-title">SBAR span extraction</text><line x1="650.0" y1="118.0" x2="650.0" y2="278.0" stroke="#d8dcef" stroke-width="1" /><text x="650.0" y="294.0" text-anchor="middle" class="tick-label">0.00</text><line x1="769.0" y1="118.0" x2="769.0" y2="278.0" stroke="#d8dcef" stroke-width="1" /><text x="769.0" y="294.0" text-anchor="middle" class="tick-label">0.50</text><line x1="888.0" y1="118.0" x2="888.0" y2="278.0" stroke="#d8dcef" stroke-width="1" /><text x="888.0" y="294.0" text-anchor="middle" class="tick-label">1.00</text><text x="769.0" y="284.0" text-anchor="middle" class="axis-label">Micro-F1</text><text x="638.0" y="144.0" text-anchor="end" class="model-label">GPT-5.2</text><rect x="650.0" y="126.0" width="121.7" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="777.7" y="135.0" class="value-label">0.51</text><rect x="650.0" y="140.0" width="179.7" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="835.7" y="149.0" class="value-label">0.76</text><rect x="650.0" y="154.0" width="141.0" height="10.0" fill="#d97706" rx="3" ry="3" /><text x="797.0" y="163.0" class="value-label">0.59</text><text x="638.0" y="188.0" text-anchor="end" class="model-label">GPT-5-nano</text><rect x="650.0" y="170.0" width="117.2" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="773.2" y="179.0" class="value-label">0.49</text><rect x="650.0" y="184.0" width="164.6" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="820.6" y="193.0" class="value-label">0.69</text><text x="638.0" y="232.0" text-anchor="end" class="model-label">MedGemma 27B</text><rect x="650.0" y="214.0" width="120.2" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="776.2" y="223.0" class="value-label">0.51</text><rect x="650.0" y="228.0" width="158.1" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="814.1" y="237.0" class="value-label">0.66</text><rect x="650.0" y="242.0" width="108.9" height="10.0" fill="#d97706" rx="3" ry="3" /><text x="764.9" y="251.0" class="value-label">0.46</text><rect x="22" y="330" width="450" height="220" rx="10" ry="10" fill="#ffffff" stroke="#d8dcef" stroke-width="1.2" /><text x="38" y="350" class="panel-title">Uncertainty span extraction</text><line x1="172.0" y1="366.0" x2="172.0" y2="526.0" stroke="#d8dcef" stroke-width="1" /><text x="172.0" y="542.0" text-anchor="middle" class="tick-label">0.00</text><line x1="291.0" y1="366.0" x2="291.0" y2="526.0" stroke="#d8dcef" stroke-width="1" /><text x="291.0" y="542.0" text-anchor="middle" class="tick-label">0.50</text><line x1="410.0" y1="366.0" x2="410.0" y2="526.0" stroke="#d8dcef" stroke-width="1" /><text x="410.0" y="542.0" text-anchor="middle" class="tick-label">1.00</text><text x="291.0" y="532.0" text-anchor="middle" class="axis-label">Micro-F1</text><text x="160.0" y="392.0" text-anchor="end" class="model-label">GPT-5.2</text><rect x="172.0" y="374.0" width="82.4" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="260.4" y="383.0" class="value-label">0.35</text><rect x="172.0" y="388.0" width="97.7" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="275.7" y="397.0" class="value-label">0.41</text><rect x="172.0" y="402.0" width="28.7" height="10.0" fill="#d97706" rx="3" ry="3" /><text x="206.7" y="411.0" class="value-label">0.12</text><text x="160.0" y="480.0" text-anchor="end" class="model-label">MedGemma 27B</text><rect x="172.0" y="462.0" width="33.1" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="211.1" y="471.0" class="value-label">0.14</text><rect x="172.0" y="476.0" width="46.6" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="224.6" y="485.0" class="value-label">0.20</text><rect x="172.0" y="490.0" width="56.4" height="10.0" fill="#d97706" rx="3" ry="3" /><text x="234.4" y="499.0" class="value-label">0.24</text><rect x="500" y="330" width="450" height="220" rx="10" ry="10" fill="#ffffff" stroke="#d8dcef" stroke-width="1.2" /><text x="516" y="350" class="panel-title">Unknown-fact span extraction</text><line x1="650.0" y1="366.0" x2="650.0" y2="526.0" stroke="#d8dcef" stroke-width="1" /><text x="650.0" y="542.0" text-anchor="middle" class="tick-label">0.00</text><line x1="769.0" y1="366.0" x2="769.0" y2="526.0" stroke="#d8dcef" stroke-width="1" /><text x="769.0" y="542.0" text-anchor="middle" class="tick-label">0.50</text><line x1="888.0" y1="366.0" x2="888.0" y2="526.0" stroke="#d8dcef" stroke-width="1" /><text x="888.0" y="542.0" text-anchor="middle" class="tick-label">1.00</text><text x="769.0" y="532.0" text-anchor="middle" class="axis-label">Micro-F1</text><text x="638.0" y="392.0" text-anchor="end" class="model-label">GPT-5.2</text><rect x="650.0" y="374.0" width="181.3" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="837.3" y="383.0" class="value-label">0.76</text><rect x="650.0" y="388.0" width="200.4" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="856.4" y="397.0" class="value-label">0.84</text><rect x="650.0" y="402.0" width="133.9" height="10.0" fill="#d97706" rx="3" ry="3" /><text x="789.9" y="411.0" class="value-label">0.56</text><text x="638.0" y="436.0" text-anchor="end" class="model-label">GPT-5-nano</text><rect x="650.0" y="418.0" width="30.7" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="686.7" y="427.0" class="value-label">0.13</text><rect x="650.0" y="432.0" width="200.4" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="856.4" y="441.0" class="value-label">0.84</text><text x="638.0" y="480.0" text-anchor="end" class="model-label">MedGemma 27B</text><rect x="650.0" y="462.0" width="102.9" height="10.0" fill="#8b95a7" rx="3" ry="3" /><text x="758.9" y="471.0" class="value-label">0.43</text><rect x="650.0" y="476.0" width="190.4" height="10.0" fill="#1f6feb" rx="3" ry="3" /><text x="846.4" y="485.0" class="value-label">0.80</text></svg>


Within-model comparison of saved evaluation runs within manuscript scope. Horizontal bars show micro-F1 for baseline, DSPy/GEPA, and LangExtract runs for each model within each task. Missing bars indicate that a given task-model-approach combination was not evaluated or was not available as a saved run; in particular, LangExtract was not evaluated for the checklist task.
:::

{{< pagebreak >}}

::: {#tbl-checklist-items}
| Category | Checklist item |
| --- | --- |
| Patient Involvement | Clinician introductions |
| Patient Involvement | Introduction of clinicians to the patient or carer |
| Patient Involvement | Invitation for the patient or carer to participate in handover |
| Identification | Verification of three patient identifiers |
| Situation | Primary diagnosis or reason for admission |
| Situation | Significant events or complications |
| Situation | Current status, including pending tests/procedures and interim plans/orders |
| Background | Relevant clinical and social history, including comorbidities |
| Background | Falls risk |
| Background | Pressure injury risk |
| Background | Allergies |
| Background | Advance care planning |
| Assessment | Observations, deterioration score, and recent escalations |
| Assessment | Pain management |
| Assessment | Devices, lines, and vascular access |
| Assessment | Critical monitoring and alarms |
| Assessment | Nutrition and dietary restrictions |
| Assessment | Fluid balance and fluid restrictions |
| Assessment | Infusions |
| Assessment | Medication chart review, including high-risk medicines |
| Assessment | Pathology results or pending investigations |
| Assessment | Mobility and use of aids |
| Assessment | Skin integrity and related interventions |
| Recommendation | Discharge plan |
| Recommendation | Critical actions required |
| Recommendation | Follow-up care plan or pathway actions |
| Recommendation | Patient or carer goals and preferences |

Checklist items used to operationalise identification of key concepts and entities in handover transcripts.
:::

{{< pagebreak >}}


::: {#tbl-uncertainty-items}
| Category | Definition / when to use | Example from handover speech |
| --- | --- | --- |
| Hedge / Probability Language | The speaker indicates partial confidence or doubt about information. | “I think ENT reviewed him.”<br>“He should be going to theatre soon.” |
| Vague / Qualitative Expression | Information is described using imprecise or subjective language. | “He looks fine now.”<br>“Seems okay.” |
| Unknown Fact / Explicit Lack of Knowledge | The speaker openly states missing knowledge or incomplete data. | “Not sure if consent’s been signed.”<br>“I don’t know his allergies.” |
| Indefinite Timing | Timing or schedule for an event is vague or lacks precision. | “Later today.”<br>“After the round.” |
| Source Uncertainty | Information relies on a second-hand or unverifiable source. | “ENT said he’s on the list.”<br>“Night nurse told me.” |
| Procedural Uncertainty | The next step in care is unclear or the plan is not explicitly stated. | “You might want to check his IV.” |
| Responsibility Uncertainty | A required task or follow-up is mentioned, but it is unclear who is responsible for performing it. | “Bloods to be checked later.”<br>“Needs review this afternoon.” |

Uncertainty categories used to support annotator identification of uncertainty-related spans in handover transcripts.
:::

::::: landscape


:::: {#tbl-sbar-optimized}

| Label | Gold | Predicted spans | Recall | Precision | Mean IoU | F1 | Delta F1 (vs baseline) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSESSMENT | 194 | 192 | 0.77 | 0.78 | 0.77 | 0.77 | +0.23 |
| BACKGROUND | 47 | 50 | 0.72 | 0.68 | 0.66 | 0.70 | +0.40 |
| RECOMMENDATION | 113 | 124 | 0.76 | 0.69 | 0.78 | 0.73 | +0.15 |
| SITUATION | 73 | 52 | 0.68 | 0.96 | 0.82 | 0.80 | +0.25 |


Per-label SBAR metrics for the best-performing GPT-5.2 DSPy/GEPA model output on the consensus evaluation partition (n=48). `Delta F1` is the difference from the corresponding GPT-5.2 baseline run on the same partition.
::::



{{< pagebreak >}}

| Checklist item | TP | FP | FN | TN | Precision | Recall | F1 |
| ------------------------------------------------ | ----: | ----: | ----: | ----: | ---------: | -------: | -----: |
| **Identification** |  |  |  |  |  |  |  |
| ID check of 3 patient identifiers | 1 | 0 | 0 | 48 | 1.00 | 1.00 | 1.00 |
| **Situation** |  |  |  |  |  |  |  |
| Primary diagnosis \| reason for admission | 45 | 2 | 0 | 2 | 0.96 | 1.00 | 0.98 |
| Current status (awaiting tests/procedures, on interim orders/plan) | 22 | 8 | 9 | 10 | 0.73 | 0.71 | 0.72 |
| Significant events or complications | 5 | 3 | 5 | 36 | 0.62 | 0.50 | 0.56 |
| **Background** |  |  |  |  |  |  |  |
| Alerts - allergies | 16 | 2 | 2 | 29 | 0.89 | 0.89 | 0.89 |
| Relevant clinical and social history \| comorbidities | 17 | 1 | 0 | 31 | 0.94 | 1.00 | 0.97 |
| Alerts - falls risk | 2 | 0 | 1 | 46 | 1.00 | 0.67 | 0.80 |
| Alerts - pressure injury risk | 2 | 0 | 1 | 46 | 1.00 | 0.67 | 0.80 |
| Advanced care planning | 1 | 0 | 0 | 48 | 1.00 | 1.00 | 1.00 |
| **Assessment** |  |  |  |  |  |  |  |
| Observations \| Q-ADDS \| recent escalations | 38 | 2 | 2 | 7 | 0.95 | 0.95 | 0.95 |
| Medication chart \| flag high risk meds | 24 | 4 | 1 | 20 | 0.86 | 0.96 | 0.91 |
| Devices \| lines \| vascular access | 23 | 3 | 0 | 23 | 0.88 | 1.00 | 0.94 |
| Mobility \| aids | 16 | 4 | 1 | 28 | 0.80 | 0.94 | 0.86 |
| Pain management | 16 | 2 | 1 | 30 | 0.89 | 0.94 | 0.91 |
| Infusions | 8 | 1 | 7 | 33 | 0.89 | 0.53 | 0.67 |
| Pathology | 14 | 5 | 1 | 29 | 0.74 | 0.93 | 0.82 |
| Nutrition \| restrictions | 14 | 6 | 0 | 29 | 0.70 | 1.00 | 0.82 |
| Fluid balance \| restrictions | 7 | 3 | 2 | 37 | 0.70 | 0.78 | 0.74 |
| Skin integrity \| interventions | 6 | 5 | 0 | 38 | 0.55 | 1.00 | 0.71 |
| Critical monitoring \| alarms | 0 | 2 | 0 | 47 | 0.00 | 0.00 | 0.00 |
| **Recommendation** |  |  |  |  |  |  |  |
| Care plan/pathway actions to follow up | 44 | 5 | 0 | 0 | 0.90 | 1.00 | 0.95 |
| Asked patient/carer about goals and preferences | 0 | 1 | 9 | 39 | 0.00 | 0.00 | 0.00 |
| Discharge plan | 4 | 1 | 0 | 44 | 0.80 | 1.00 | 0.89 |
| Critical actions required | 1 | 4 | 0 | 44 | 0.20 | 1.00 | 0.33 |
| **Patient Involvement** |  |  |  |  |  |  |  |
| Introduction of clinicians involved in handover to patient/carer | 20 | 8 | 0 | 21 | 0.71 | 1.00 | 0.83 |
| Invitation for patient/carer to participate in handover | 15 | 8 | 1 | 25 | 0.65 | 0.94 | 0.77 |

: Grouped per-label checklist performance for the best-performing GPT-5.2 DSPy/GEPA model on the consensus evaluation partition (`n=49`). {#tbl-checklist-grouped tbl-colwidths="[55,6,6,6,6,8,8,8]"}

*Legend:* TP = true positives; FP = false positives; FN = false negatives; TN = true negatives; F1 = harmonic mean of precision and recall; ID = identification; Q-ADDS = Queensland Adult Deterioration Detection System.


:::::

