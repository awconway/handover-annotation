---
title: "Large language models for structured information extraction in artificial intelligence-assisted clinical handover"

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

Two registered nurses independently annotated a dataset of 203 synthetic handover transcripts to produce consensus labels for each information extraction task. Tasks included: 1) labelling spans of text into SBAR (Situation, Background, Assessment, Recommendation) categories; 2)  content detectionjudging if specific pieces of information were communicated; and 3) labelling spans of text that communicated information using uncertain terms that included a sub-task for identifying unknown facts. Baseline and Genetic-Pareto (GEPA) optimised prompts were compared for GPT-5.2, GPT-5-nano, and MedGemma 27B large langage models. Additionally, the LangExtract framework was evaluated for span-extraction tasks. Content detection was scored using micro-, macro-, and support-weighted precision, recall, and F1; span tasks were scored using same-label quote matching, matched-span precision/recall/F1, and mean intersection-over-union for boundary agreement.

### Results

In matched GPT-5.2 comparisons, prompt optimisation increased point-estimate micro-F1 for content detection, SBAR span extraction, uncertainty-span extraction, and unknown-fact extraction. The GPT-5.2 optimised model achieved micro-F1 0\.85 \(95% CI 0\.83\-0\.88\) for content detection. For SBAR span extraction, GPT-5.2 with GEPA prompt optimisation achieved micro-F1 0\.76 \(95% CI 0\.72\-0\.79\). Broad uncertainty-span extraction remained comparatively weak (micro-F1 0\.41 \(95% CI 0\.33\-0\.48\)), whereas explicit unknown-fact extraction was stronger (micro-F1 0\.84 \(95% CI 0\.63\-1\.00\)). Prompt optimisation outperformed the LangExtract approach across each task.

### Conclusion

Prompt-optimisation enhanced the performance of large language models for tasks that can be used in real-time clinical handover support. Broad uncertainty detection was not sufficiently accurate to be used in clinical practice. For most tasks, the smaller GPT-5-nano model performed competitively with the larger GPT-5.2, suggesting that cost-effective models may be viable for certain applications. Future work should explore integration of these capabilities into clinical workflows and evaluate their impact on communication quality and patient outcomes.

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

Sampling uncertainty was summarised with 95% confidence intervals calculated using non-parametric bootstrap resampling of evaluation transcripts. For each result, we resampled transcripts with replacement 2,000 times using a fixed random seed and recalculated the relevant metric from the pooled counts in each resample. Confidence limits are reported as the 2.5th and 97.5th percentiles of the bootstrap distribution. For checklist items, accuracy was calculated as (true positives + true negatives) divided by all evaluated transcripts for that item. Confidence intervals for precision, recall, F1, macro-F1, support-weighted F1, and mean IoU were calculated by applying the same transcript-level bootstrap procedure and recomputing each statistic within each resample.


## Results {#sec-results}

### Key findings {#sec-results-summary}

Within-model comparisons showed consistent point-estimate gains with DSPy/GEPA over matched baselines wherever both were available. For GPT-5.2, micro-F1 increased from 0\.77 \(95% CI 0\.74\-0\.80\) to 0\.85 \(95% CI 0\.83\-0\.88\) for checklist prediction, from 0\.51 \(95% CI 0\.47\-0\.55\) to 0\.76 \(95% CI 0\.72\-0\.79\) for SBAR span extraction, from 0\.35 \(95% CI 0\.28\-0\.41\) to 0\.41 \(95% CI 0\.33\-0\.48\) for uncertainty span extraction, and from 0\.76 \(95% CI 0\.50\-1\.00\) to 0\.84 \(95% CI 0\.63\-1\.00\) for unknown-fact extraction. Across span tasks, LangExtract generally performed below the corresponding best DSPy/GEPA configuration.

### Comparative performance across tasks and prompting approaches {#sec-results-overview}

@fig-results-overview summarizes matched within-model comparisons across tasks for the three models evaluated in the main study. 


### SBAR span extraction {#sec-sbar-optimized}

Among SBAR configurations, the highest overall score was achieved by DSPy/GEPA-optimised GPT-5.2 (micro-F1 0\.76 \(95% CI 0\.72\-0\.79\)), followed by DSPy/GEPA-optimised GPT-5-nano (micro-F1 0\.69 \(95% CI 0\.66\-0\.72\)) and LangExtract GPT-5.2 (micro-F1 0\.59 \(95% CI 0\.56\-0\.62\)). @tbl-sbar-optimized provides label-level results for the best-performing GPT-5.2 DSPy/GEPA SBAR configuration.


Within this GPT-5.2 SBAR comparison, macro-precision increased from 0\.41 \(95% CI 0\.37\-0\.44\) to 0\.78 \(95% CI 0\.73\-0\.82\), macro-recall from 0\.69 \(95% CI 0\.63\-0\.75\) to 0\.73 \(95% CI 0\.69\-0\.78\), and macro-F1 from 0\.49 \(95% CI 0\.46\-0\.53\) to 0\.75 \(95% CI 0\.71\-0\.79\). Span-boundary agreement among matched predictions was strongest for SITUATION and RECOMMENDATION, as shown by the label-level mean IoU estimates in @tbl-sbar-optimized.

### Checklist task {#sec-checklist}

For checklist prediction, the best overall result was achieved by DSPy/GEPA-optimised GPT-5.2 (micro-F1 0\.85 \(95% CI 0\.83\-0\.88\), macro-F1 0\.73 \(95% CI 0\.63\-0\.76\), support-weighted F1 0\.85 \(95% CI 0\.82\-0\.88\)). DSPy/GEPA-optimised GPT-5-nano also performed competitively (micro-F1 0\.81 \(95% CI 0\.78\-0\.84\)), while MedGemma 27B reached micro-F1 0\.76 \(95% CI 0\.74\-0\.79\)). @tbl-checklist-grouped presents grouped per-label estimates for accuracy, precision, recall, and F1 for the best-performing GPT-5.2 checklist model. Confidence intervals are not provided for labels with few positive examples in the test set. 2 low-support labels still had F1 equal to zero.

### Uncertainty and unknown-fact span extraction {#sec-uncertainty-results}

Broad uncertainty-span extraction remained the weakest task. The best DSPy/GEPA run used GPT\-5\.2 and achieved precision 0\.32 \(95% CI 0\.26\-0\.39\), recall 0\.56 \(95% CI 0\.44\-0\.67\), micro-F1 0\.41 \(95% CI 0\.33\-0\.48\), and mean IoU 0\.83 \(95% CI 0\.77\-0\.90\). This was only a modest improvement over the best baseline run (micro-F1 0\.35 \(95% CI 0\.28\-0\.41\)), and the best LangExtract uncertainty run remained low (micro-F1 0\.24 \(95% CI 0\.17\-0\.31\)).

The narrower unknown-fact sub-task performed substantially better. DSPy/GEPA reached micro-F1 0\.84 \(95% CI 0\.63\-1\.00\) with GPT\-5\.2, compared with the best baseline micro-F1 of 0\.76 \(95% CI 0\.50\-1\.00\) and the LangExtract GPT-5.2 micro-F1 of 0\.56 \(95% CI 0\.29\-0\.72\). These results suggest that explicitly stated information gaps were more tractable than the broader set of hedged, vague, source-dependent, procedural, and responsibility-related uncertainty spans. @tbl-uncertainty-task-results summarises the uncertainty and unknown-fact span results used for these comparisons.


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

![Within-model comparison of saved evaluation runs within manuscript scope. Horizontal bars show micro-F1 for baseline, DSPy/GEPA, and LangExtract runs for each model within each task. Missing bars indicate that a given task-model-approach combination was not evaluated or was not available as a saved run; in particular, LangExtract was not evaluated for the checklist task.](manuscript-assets/fig-results-overview.png){#fig-results-overview width=6.8in fig-alt="Four-panel grouped horizontal bar chart showing micro-F1 for checklist, SBAR, uncertainty, and unknown-fact tasks by model and prompting approach."}

{{< pagebreak >}}

::: {#tbl-checklist-items tbl-colwidths="[30,70]"}
| Category | Checklist item |
| :--- | :--- |
| Patient involvement | Clinician introductions |
|  | Introduction of clinicians to the patient or carer |
|  | Invitation for the patient or carer to participate in handover |
| Identification | Verification of three patient identifiers |
| Situation | Primary diagnosis or reason for admission |
|  | Significant events or complications |
|  | Current status, including pending tests/procedures and interim plans/orders |
| Background | Relevant clinical and social history, including comorbidities |
|  | Falls risk |
|  | Pressure injury risk |
|  | Allergies |
|  | Advance care planning |
| Assessment | Observations, deterioration score, and recent escalations |
|  | Pain management |
|  | Devices, lines, and vascular access |
|  | Critical monitoring and alarms |
|  | Nutrition and dietary restrictions |
|  | Fluid balance and fluid restrictions |
|  | Infusions |
|  | Medication chart review, including high-risk medicines |
|  | Pathology results or pending investigations |
|  | Mobility and use of aids |
|  | Skin integrity and related interventions |
| Recommendation | Discharge plan |
|  | Critical actions required |
|  | Follow-up care plan or pathway actions |
|  | Patient or carer goals and preferences |

Checklist items used to operationalise identification of key concepts and entities in handover transcripts. Blank category cells indicate continuation of the preceding category.
:::

{{< pagebreak >}}


::: {#tbl-uncertainty-items tbl-colwidths="[22,43,35]"}
| Category | Definition / when to use | Example from handover speech |
| :--- | :--- | :--- |
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


:::: {#tbl-sbar-optimized tbl-colwidths="[18,8,13,14,14,15,18]"}

| Label | Gold | Predicted spans | Recall | Precision | Mean IoU | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSESSMENT | 194 | 192 | 0.77 (0.72-0.82) | 0.78 (0.72-0.83) | 0.77 (0.72-0.81) | 0.77 (0.73-0.81) |
| BACKGROUND | 47 | 50 | 0.72 (0.61-0.84) | 0.68 (0.57-0.79) | 0.66 (0.56-0.75) | 0.70 (0.60-0.80) |
| RECOMMENDATION | 113 | 124 | 0.76 (0.69-0.84) | 0.69 (0.62-0.78) | 0.78 (0.73-0.83) | 0.73 (0.67-0.78) |
| SITUATION | 73 | 52 | 0.68 (0.57-0.81) | 0.96 (0.88-1.00) | 0.82 (0.75-0.88) | 0.80 (0.72-0.89) |


Per-label SBAR metrics for the best-performing GPT-5.2 DSPy/GEPA model output on the consensus evaluation partition (n=48). Metric cells show point estimate (95% CI).
::::

*Legend:* Gold = reference spans; predicted spans = model-generated spans mapped back to the transcript; IoU = intersection over union for matched span boundaries; F1 = harmonic mean of precision and recall.



{{< pagebreak >}}

| Checklist item | Accuracy | Precision | Recall | F1 |
| -------------------------------------------- | -------------: | -------------: | -------------: | -------------: |
| **Identification** |  |  |  |  |
| ID check of 3 patient identifiers | 1.00 | 1.00 | 1.00 | 1.00 |
| **Situation** |  |  |  |  |
| Primary diagnosis \| reason for admission | 0.96 (0.90-1.00) | 0.96 (0.89-1.00) | 1.00 | 0.98 (0.94-1.00) |
| Current status (awaiting tests/procedures, on interim orders/plan) | 0.65 (0.51-0.78) | 0.73 (0.57-0.88) | 0.71 (0.55-0.87) | 0.72 (0.58-0.84) |
| Significant events or complications | 0.84 (0.73-0.94) | 0.62 (0.25-1.00) | 0.50 (0.17-0.83) | 0.56 (0.20-0.80) |
| **Background** |  |  |  |  |
| Alerts - allergies | 0.92 (0.84-0.98) | 0.89 (0.73-1.00) | 0.89 (0.71-1.00) | 0.89 (0.76-0.98) |
| Relevant clinical and social history \| comorbidities | 0.98 (0.94-1.00) | 0.94 (0.81-1.00) | 1.00 | 0.97 (0.90-1.00) |
| Alerts - falls risk | 0.98 (0.94-1.00) | 1.00 | 0.67 | 0.80 |
| Alerts - pressure injury risk | 0.98 (0.94-1.00) | 1.00 | 0.67 | 0.80 |
| Advanced care planning | 1.00 | 1.00 | 1.00 | 1.00 |
| **Assessment** |  |  |  |  |
| Observations \| Q-ADDS \| recent escalations | 0.92 (0.84-0.98) | 0.95 (0.87-1.00) | 0.95 (0.87-1.00) | 0.95 (0.89-0.99) |
| Medication chart \| flag high risk meds | 0.90 (0.82-0.98) | 0.86 (0.71-0.97) | 0.96 (0.87-1.00) | 0.91 (0.81-0.98) |
| Devices \| lines \| vascular access | 0.94 (0.88-1.00) | 0.88 (0.75-1.00) | 1.00 | 0.94 (0.86-1.00) |
| Mobility \| aids | 0.90 (0.80-0.98) | 0.80 (0.61-0.95) | 0.94 (0.80-1.00) | 0.86 (0.72-0.97) |
| Pain management | 0.94 (0.86-1.00) | 0.89 (0.71-1.00) | 0.94 (0.81-1.00) | 0.91 (0.80-1.00) |
| Infusions | 0.84 (0.73-0.94) | 0.89 (0.67-1.00) | 0.53 (0.27-0.79) | 0.67 (0.40-0.86) |
| Pathology | 0.88 (0.78-0.96) | 0.74 (0.53-0.93) | 0.93 (0.79-1.00) | 0.82 (0.67-0.94) |
| Nutrition \| restrictions | 0.88 (0.78-0.96) | 0.70 (0.50-0.89) | 1.00 | 0.82 (0.67-0.94) |
| Fluid balance \| restrictions | 0.90 (0.80-0.98) | 0.70 (0.38-1.00) | 0.78 (0.44-1.00) | 0.74 (0.44-0.94) |
| Skin integrity \| interventions | 0.90 (0.80-0.98) | 0.55 (0.22-0.83) | 1.00 | 0.71 (0.36-0.91) |
| Critical monitoring \| alarms | 0.96 (0.90-1.00) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) |
| **Recommendation** |  |  |  |  |
| Care plan/pathway actions to follow up | 0.90 (0.80-0.98) | 0.90 (0.80-0.98) | 1.00 | 0.95 (0.89-0.99) |
| Asked patient/carer about goals and preferences | 0.80 (0.67-0.90) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) |
| Discharge plan | 0.98 (0.94-1.00) | 0.80 (0.33-1.00) | 1.00 | 0.89 (0.50-1.00) |
| Critical actions required | 0.92 (0.84-0.98) | 0.20 (0.00-0.67) | 1.00 | 0.33 (0.00-0.80) |
| **Patient Involvement** |  |  |  |  |
| Introduction of clinicians involved in handover to patient/carer | 0.84 (0.73-0.94) | 0.71 (0.55-0.87) | 1.00 | 0.83 (0.71-0.93) |
| Invitation for patient/carer to participate in handover | 0.82 (0.69-0.92) | 0.65 (0.44-0.83) | 0.94 (0.78-1.00) | 0.77 (0.59-0.89) |

: Grouped per-label checklist performance for the best-performing GPT-5.2 DSPy/GEPA model on the consensus evaluation partition (`n=49`). Metric cells show point estimate (95% CI) unless otherwise noted. {#tbl-checklist-grouped tbl-colwidths="[44,14,14,14,14]"}

*Legend:* Accuracy = (true positives + true negatives) / all evaluated transcripts for that checklist item; F1 = harmonic mean of precision and recall. Confidence intervals are not provided for labels with few positive examples in the test set; ID = identification; Q-ADDS = Queensland Adult Deterioration Detection System.


{{< pagebreak >}}

::: {#tbl-uncertainty-task-results tbl-colwidths="[16,15,15,13,13,13,15]"}

| Task | Model | Approach | Precision | Recall | F1 | Mean IoU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Uncertainty | GPT-5.2 | Baseline | 0.24 (0.19-0.29) | 0.65 (0.53-0.75) | 0.35 (0.28-0.41) | 0.62 (0.53-0.70) |
|  |  | DSPy/GEPA | 0.32 (0.26-0.39) | 0.56 (0.44-0.67) | 0.41 (0.33-0.48) | 0.83 (0.77-0.90) |
|  |  | LangExtract | 0.35 (0.21-0.55) | 0.07 (0.03-0.13) | 0.12 (0.05-0.20) | 0.35 (0.28-0.43) |
|  | MedGemma 27B | Baseline | 0.12 (0.07-0.18) | 0.17 (0.10-0.23) | 0.14 (0.08-0.20) | 0.56 (0.39-0.70) |
|  |  | DSPy/GEPA | 0.19 (0.12-0.31) | 0.20 (0.13-0.28) | 0.20 (0.13-0.27) | 0.72 (0.55-0.88) |
|  |  | LangExtract | 0.22 (0.15-0.29) | 0.26 (0.18-0.36) | 0.24 (0.17-0.31) | 0.69 (0.57-0.80) |
| Unknown fact | GPT-5.2 | Baseline | 0.73 (0.44-1.00) | 0.80 (0.56-1.00) | 0.76 (0.50-1.00) | 0.86 (0.66-0.99) |
|  |  | DSPy/GEPA | 0.89 (0.71-1.00) | 0.80 (0.56-1.00) | 0.84 (0.63-1.00) | 0.92 (0.74-0.99) |
|  |  | LangExtract | 0.41 (0.17-0.59) | 0.90 (0.79-1.00) | 0.56 (0.29-0.72) | 0.68 (0.45-0.84) |
|  | GPT-5-nano | Baseline | 0.08 (0.00-0.17) | 0.40 (0.00-1.00) | 0.13 (0.00-0.27) | 0.74 (0.00-0.97) |
|  |  | DSPy/GEPA | 0.89 (0.71-1.00) | 0.80 (0.56-1.00) | 0.84 (0.63-1.00) | 0.89 (0.79-0.97) |
|  | MedGemma 27B | Baseline | 0.30 (0.09-0.48) | 0.80 (0.56-1.00) | 0.43 (0.16-0.61) | 0.90 (0.72-0.98) |
|  |  | DSPy/GEPA | 0.80 (0.67-1.00) | 0.80 (0.56-1.00) | 0.80 (0.63-1.00) | 0.90 (0.72-0.98) |


Uncertainty and unknown-fact span extraction results on the consensus evaluation partition (`n=37`). Metric cells show point estimate (95% CI). Blank task or model cells indicate continuation of the preceding group.
:::

*Legend:* IoU = intersection over union for matched span boundaries; F1 = harmonic mean of precision and recall.

:::::

