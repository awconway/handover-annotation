---
title: "Artificial Intelligence–Based Structured Information Extraction From Synthetic Nursing Handover Transcripts: Comparative Evaluation of Large Language Models"

authors:
  - name: Aaron Conway
    affiliations:
      - ref: qutcht
      - ref: qutnursing
      - ref: tpch
    corresponding: true
    email: aaron.conway@qut.edu.au
    orcid: 0000-0002-9583-8636
  - name: Adriana Hada
    affiliations:
      - ref: tpch
    orcid: 0000-0001-9794-6502
  - name: Jessica Schluter
    affiliations:
      - ref: tpch
  - name: Hui (Grace) Xu
    affiliations:
      - ref: qutnursing
      - ref: rbwh
    orcid: 0000-0002-3421-4176
  - name: Dan Lowden
    affiliations:
      - ref: caboolture
  - name: Tim Miller
    affiliations:
      - ref: uqcomp
    orcid: 0000-0003-4908-6063
  - name: Ken Donald
    affiliations:
      - ref: griffith
  - name: Andrew Teodorczuk
    affiliations:
      - ref: uqmed
      - ref: ranzcp
      - ref: tpch
    orcid: 0000-0003-0802-718X

affiliations:
  - id: qutcht
    name: Centre for Healthcare Transformation, Queensland University of Technology, Brisbane, Australia
  - id: qutnursing
    name: School of Nursing, Queensland University of Technology, Brisbane, Australia
  - id: tpch
    name: The Prince Charles Hospital, Metro North Health, Brisbane, Australia
  - id: rbwh
    name: Royal Brisbane and Women's Hospital, Brisbane, Australia
  - id: caboolture
    name: Caboolture Hospital, Brisbane, Australia
  - id: uqcomp
    name: School of Electrical Engineering and Computer Science, The University of Queensland, Brisbane, Australia
  - id: griffith
    name: School of Medicine & Dentistry, Griffith University, Gold Coast, Queensland, Australia
  - id: uqmed
    name: The University of Queensland Northside Clinical Unit, The University of Queensland, Brisbane, Australia
  - id: ranzcp
    name: Royal Australian and New Zealand College of Psychiatrists, Melbourne, Australia


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


### Background

Clinical handover is the process during which responsibility and accountability for care are transferred between clinicians. Artificial intelligence has the potential to improve the reliability and completeness of clinical handover by helping clinicians detect predefined content areas that have been communicated, identify explicit information gaps, and prompt clarification before responsibility is transferred.

### Objective

This study evaluated the performance of several large language models and prompt optimisation strategies for structured information extraction of synthetic nursing handover transcripts.

### Methods

Two registered nurses independently annotated a dataset of 203 synthetic handover transcripts to produce consensus labels for information extraction tasks. Tasks included: 1) labelling spans of text into SBAR (Situation, Background, Assessment, Recommendation) categories; 2) content detection to determine if specific pieces of information were communicated; and 3) labelling spans of text that communicated information using uncertain terms that included a sub-task for identifying unknown facts. Baseline and Genetic-Pareto (GEPA) optimised prompts were compared for GPT-5.2, GPT-5-nano, and MedGemma 27B large language models. Additionally, the LangExtract framework was evaluated for span-extraction tasks.

### Results

The GPT-5.2 optimised model achieved micro-F1 0\.85 \(95% CI 0\.83\-0\.88\) for content detection, an absolute improvement of \+0\.08 compared with the matched baseline. GPT-5-nano also performed better after optimisation for content detection (micro-F1 0\.81 \(95% CI 0\.78\-0\.84\)), suggesting that this structured task was not limited to the highest-capacity model. For SBAR span extraction, GPT-5.2 with prompt optimisation achieved micro-F1 0\.76 \(95% CI 0\.72\-0\.79\), improving by \+0\.24 compared with baseline and exceeding LangExtract; GPT-5-nano also improved to micro-F1 0\.69 \(95% CI 0\.66\-0\.72\). Broad uncertainty-span extraction remained comparatively weak despite prompt optimisation (micro-F1 0\.41 \(95% CI 0\.33\-0\.48\); absolute improvement \+0\.06). In contrast, explicit unknown-fact extraction was more accurate with GPT-5.2 (micro-F1 0\.84 \(95% CI 0\.63\-1\.00\)), GPT-5-nano (micro-F1 0\.84 \(95% CI 0\.63\-1\.00\)), and MedGemma 27B (micro-F1 0\.80 \(95% CI 0\.63\-1\.00\)). Genetic-Pareto optimised prompts outperformed the LangExtract approach across each span-extraction task.

### Conclusions

Prompt optimisation improved matched-model point estimates, with the highest performance observed for predefined content detection and SBAR span extraction. Broad uncertainty extraction remained less accurate than the narrower unknown-fact task. These technical results do not establish clinical effectiveness, safety, or readiness for real-time use. Validation using authentic nursing handover communication and prospective evaluation in clinical workflows are required before clinical application.


## Keywords {#sec-keywords}

artificial intelligence; clinical handover; nursing; patient handoff; natural language processing; large language models; prompt engineering; information extraction; patient safety; machine learning

{{< pagebreak >}}


## Introduction {#sec-introduction}

Clinical handover is the process during which responsibility and accountability for some or all aspects of a patient's care are transferred to another clinician or clinical team [@acsqhc_nsqhs_comm_at_handover]. In nursing, shift-to-shift and transfer handovers support continuity of surveillance, care priorities, pending actions, and the inclusion of patient and family concerns. Inaccurate, incomplete, or misinterpreted communication at this transition can contribute to delays, duplicated work, and preventable harm [@ong2011handoff_failures; @manias2016handover_perspectives]. International and Australian patient-safety standards therefore prioritise structured clinical handover, particularly at shift changes, transfers, and discharge [@who2007patient-handover; @acsqhc_nsqhs_comm_at_handover].

Structured handover processes aim to standardise minimum content and the format of exchange so that critical information is predictably conveyed, acted upon, and auditable. In Australia, the Communicating for Safety Standard emphasises structured clinical handover while retaining opportunities for questions, clarification, and confirmation [@acsqhc_nsqhs_comm_at_handover; @hada2021nursing_handover]. One widely used framework is SBAR (Situation, Background, Assessment, Recommendation), which organises clinician-to-clinician handover content into four information categories [@yung2023handover_mnemonics]. Recent systematic review evidence suggests that structured handoff protocols may improve some safety outcomes, but the certainty and implementation fidelity vary by protocol and setting; evidence specific to SBAR remains low certainty [@mccarthy2025structured_handoff].

Even where structured tools are mandated or encouraged, their enactment in nursing handovers remains shaped by ward-level organisational and cultural conditions, time demands, interruptions, and the need to communicate patient-specific information for continuity of care [@moyo2024structured_handover_frameworks; @chien2024ward_handover_intervention]. This variability is amplified in bedside nursing handover where patient/family involvement is increasingly emphasised, yet research highlights tensions between standardisation (predictability) and tailoring (patient-centredness), along with barriers related to confidentiality and clinician concerns [@tobiano2018patient_participation_bedside_handover; @anshasi2024bedside_handover]. Reviews of handover mnemonics similarly emphasise local validation, clarification, and readback rather than assuming that every element is universally relevant [@yung2023handover_mnemonics].

Traditional supervised information-extraction methods can be highly effective for stable tasks with sufficiently large, task-specific labelled datasets. However, there are several potential advantages of using LLMs in this context. For example, the same instruction-driven model can be configured for heterogeneous, context-dependent tasks. This may be useful where annotated nursing handover data are limited and communication is conversational and non-linear. Prior work has demonstrated few-shot clinical information extraction with LLMs can be accurate, while also showing that performance depends materially on prompt design and task framing [@agrawal-etal-2022-large; @sivarajkumar2024prompting_strategies_clinical_nlp].



Recent advances in large language models (LLMs) create an opportunity to support clinical handover by analysing information as it is communicated. Other studies have examined AI-based approaches to support clinical handover, although evaluations of these technologies remain limited [@AghaMirSalim2025technological_solutions_inpatient_handover]. For example, a recent multi-hospital study used an LLM to pre-generate content to support the preparation for handover [@chen2026llm_nursing_handover]. An alternative application of AI yet to be investigated is to support the verbal clinician-to-clinician exchange itself. This spoken exchange remains central to transferring responsibility of care in clinical settings and provides opportunities to question, clarify, and confirm information [@acsqhc_nsqhs_comm_at_handover; @chien2024ward_handover_intervention]. Accurate extraction of structured information from the spoken exchange is a prerequisite for developing this form of communication support. The aim of this study was to evaluate the accuracy of LLM structured information extraction from synthetic clinical handover transcripts.

## Methods {#sec-methods}

### Study Design

This study used a model-evaluation design, corresponding to the "LLM evaluation" research-design category in TRIPOD-LLM [@gallifant2025tripod]. This category covers studies that assess existing LLMs for their accuracy or suitability for a specific healthcare task. We compared selected LLMs and prompt configurations on predefined multilabel classification and span-extraction tasks using completed synthetic nursing handover transcripts and registered nurse annotations. The study was not an evaluation in a healthcare setting because it did not test workflow integration or clinical, administrative, or workforce outcomes.

Tasks related to clinical handover communication that were considered as potentially augmentable with AI assistance were identified by the researchers using a co-design process with clinicians and consumers, which will be reported separately. The tasks were:

- Extracting spans of text from transcripts that aligned with the SBAR (Situation, Background, Assessment, Recommendation) framework for structuring clinical handover. In this study, SBAR span extraction was operationalised as a sequence-labeling task to identify contiguous spans of text in handover transcripts that corresponded to each SBAR category. This task evaluated whether models could map conversational handover text to SBAR-labelled spans, rather than whether the transcripts themselves followed a clean sequential SBAR structure.

- Identifying if key elements were communicated in handover transcripts as a content detection task. In this study, this task was operationalised as a checklist of items that are recommended to be addressed during nursing clinical handover, which were developed as part of quality-improvement processes at the researchers' institution. The checklist items are summarised in @tbl-checklist-items.
- Identifying spans of text in transcripts that were communicated using uncertain terms. In this study, this task was operationalised as span annotation of utterances during handover that conveyed incomplete knowledge, vague or hedged wording, imprecise timing, second-hand sourcing, unclear procedures, or unclear responsibility for follow-up actions. These forms of uncertainty were treated as potentially clinically important because they may indicate information that requires clarification or verification by the receiving clinician. The uncertainty categories and example guidance provided to annotators are summarised in @tbl-uncertainty-items. 

### Data sources

We used the publicly available NICTA Synthetic Nursing Handover Dataset, which contains synthetic recordings of clinical handovers delivered by a registered nurse based on patient profiles with cardiovascular, neurological, renal, and respiratory conditions [@suominen2015benchmarking]. In this dataset, handover monologues were generated from comprehensive patient profiles that included information such as the patient’s name, age, admission history, inpatient duration, and the familiarity between the nurses giving and receiving the handover. The nurse was instructed to simulate a bedside shift-to-shift handover within a medical ward setting [@suominen2015benchmarking].

For our study, we used 100 handover samples from the training partition of the NICTA dataset. First, audio recordings from the NICTA dataset were transcribed using the OpenAI Whisper speech-to-text model. Second, three videos depicting conversational nursing shift-to-shift handovers were transcribed in a similar manner to provide examples of interactive handover dialogue. These videos were developed at the authors’ institution for educational purposes to demonstrate best practices for clinical handovers that are used in the undergraduate nursing program. Transcripts from the educational videos were then used as few-shot examples within the DSPy framework[@khattab2023dspy] using the `BootstrapFewShot` optimizer to guide the transformation of the 100 NICTA monologue transcripts into two-sided conversational handovers so that the dataset would better reflect real-world contemporary clinical handover interactions. To broaden the range of clinical contexts represented in the dataset, we further synthesised 103 additional handover examples using the GPT-5 model interactively in a chat interface. These scenarios included inter- and intra-hospital transfers, post-procedural handovers, emergency department transitions, and handovers that involved patients with complex mental health care needs. The final dataset comprised 203 synthetic handover transcripts used for subsequent annotation and model development. 

### Ethical Considerations

Ethics approval was not sought because this was not human research as defined by the National Statement on Ethical Conduct in Human Research [@NHMRC2025NationalStatement]. The study involved only synthetic handover transcripts generated for model evaluation, with no recruitment, observation or testing of human participants, and no use of patient, clinician, clinical-record, personal, identifiable or potentially re-identifiable data. Accordingly, the study did not require human ethics review, and a formal exemption from ethics review was not applicable.

### Annotation

Two annotators, who are experienced clinically active Registered Nurses, independently annotated transcripts using a structured rubric. Labelled annotations that met consensus between reviewers were used as the reference labels for downstream model development and evaluation.

Annotation was performed using Prodigy annotation software using a custom interface that presented each transcript in three components [@montani2018prodigy]. First, annotators highlighted relevant spans of text and assigned labels corresponding to the SBAR framework together with markers of communicative uncertainty, including vagueness, hedging, unknown facts, indefinite timing, source uncertainty, procedural uncertainty, and uncertainty regarding responsibility. Overlapping span labels were permitted where a passage served more than one communicative function. Second, annotators completed a multiple-response checklist indicating whether predefined handover elements were present in the transcript.

Both annotators reviewed all transcripts independently within the same annotation environment and were supported by written guidance and examples to promote consistent interpretation of the coding framework. For creation of the reference standard, a consensus dataset was derived by retaining only those span annotations and checklist items for which both annotators agreed. In practical terms, this meant that only text segments assigned the same label by both reviewers, and only checklist items selected by both reviewers, were carried forward for downstream model development and evaluation.

### Large language model prompt optimisation methods

Annotated handover transcripts were first partitioned deterministically into optimisation (75%) and evaluation (25%) subsets using a fixed random split. Prompt optimisation was performed within DSPy [@khattab2023dspy], which is a Python framework that can be used for prompt optimisation using feedback from model outputs to improve task performance. For each task, we defined a task-specific DSPy signature that specified the transcript as input and a constrained structured output. Checklist prediction was formulated as multilabel classification, where the task was to return a list of the items from the checklist that were covered in the transcript. The SBAR and uncertainty-related tasks were formulated as span extraction requiring the model to return verbatim text segments from the transcript together with the appropriate label. The uncertainty-related tasks were further subdivided into a broad uncertainty-span extraction task (which included all uncertainty categories) and a more specific unknown-fact extraction task (which included only spans labelled as unknown facts).

Across the baseline DSPy evaluations, GEPA-based DSPy optimisation experiments, and LangExtract experiments, we purposively selected 3 underlying language models with different deployment profiles: OpenAI GPT-5.2 as the higher-capacity proprietary model, OpenAI GPT-5-nano as a smaller lower-cost proprietary model [@singh2025openaigpt5card], and MedGemma 27B as a 27-billion-parameter open-weight model developed for medical tasks [@sellergren2025medgemmatechnicalreport]. This panel was intended to examine matched prompt effects across contrasting model profiles, not to provide an exhaustive leaderboard of all contemporary LLMs. For DSPy baseline and GEPA runs, the same task model was used before and after optimisation so that differences reflected the prompt configuration rather than a change in the underlying model. GPT-5.2 and GPT-5-nano were accessed through cloud-hosted OpenAI API endpoints; MedGemma 27B was run locally on institutional high-performance computing infrastructure.

GEPA was used to optimise the task prompts [@agrawal2025gepa]. It iteratively evaluated candidate task instructions, combined numeric task scores with natural-language error feedback, and used GPT-5.2 as a separate reflection model to propose revisions. Optimisation used 576 scoring calls. Checklist optimisation targeted multilabel agreement, while span-task optimisation rewarded same-label text overlap using IoU so that closer boundaries received higher scores. A transcript with neither a reference span nor a predicted span for a target label was treated as a correct negative. The final compiled prompts were fixed before evaluation on the held-out partition. For reporting, span detection performance and boundary overlap were presented separately.

In addition to DSPy prompt optimisations, we conducted separate evaluations using the LangExtract framework, as a prompt-based few-shot structured information extraction approach for the SBAR, uncertainty-span, and unknown-fact span tasks [@goel_2026_langextract]. These experiments used task-specific prompt descriptions together with annotated in-context examples derived from the reference data. We used 10 annotated examples from the training partition as few-shot exemplars, and inference was then performed on the full held-out test partition for each task.

### Data analysis

Inter-rater agreement was assessed with Cohen's kappa and mean intersection over union (IoU) among matched spans. Agreement confidence intervals were calculated with 2,000 bootstrap resamples of transcript pairs using a fixed random seed.

Performance was measured for the checklist content detection task at the level of individual labels with counts of true positives, false positives, false negatives, and true negatives, together with precision, recall, and F1. Aggregate performance was summarised using micro-averaged (pooled), macro-averaged (unweighted mean), and support-weighted precision, recall, and F1 across labels.

For span-extraction tasks, precision, recall, and F1 were calculated from predicted spans as binary detection measures, so that these statistics reflected the model's ability to identify the correct labelled spans. Span-boundary agreement was reported separately using the mean intersection over union (IoU) across matched pairs. We additionally calculated per-label descriptive metrics including the number of reference spans, the number of predicted spans, matched-span precision, recall, F1, and mean IoU.

Sampling uncertainty was summarised with 95% confidence intervals calculated using non-parametric bootstrap resampling. For each result, we resampled transcripts with replacement 2,000 times using a fixed random seed and recalculated the relevant metric from the pooled counts in each resample. Confidence limits are reported as the 2.5th and 97.5th percentiles of the bootstrap distribution.


## Results {#sec-results}

### Pre-consensus inter-rater agreement {#sec-inter-rater-agreement}

Cohen's kappa was 0.75 (95% CI 0.73-0.76) for checklist decisions, 0.70 (95% CI 0.68-0.72) for pooled SBAR token-by-label decisions, 0.12 (95% CI 0.08-0.15) for broad uncertainty, and 0.40 (95% CI 0.18-0.64) for unknown facts. Among overlapping same-label spans, mean IoU was 0.86 (95% CI 0.85-0.87) for SBAR, 0.71 (95% CI 0.61-0.81) for broad uncertainty, and 0.78 (95% CI 0.56-0.98) for unknown facts. The relatively high matched-span IoU indicates similar boundaries when both nurses identified the same span type, with disagreement in uncertainty annotations arising mainly over whether and how to label an expression.

### Key findings {#sec-results-summary}

Within-model comparisons showed consistent point-estimate gains with DSPy/GEPA over matched unoptimised-prompt baselines. For GPT-5.2, micro-F1 increased from 0\.77 \(95% CI 0\.74\-0\.80\) to 0\.85 \(95% CI 0\.83\-0\.88\) for checklist prediction, from 0\.51 \(95% CI 0\.47\-0\.55\) to 0\.76 \(95% CI 0\.72\-0\.79\) for SBAR span extraction, from 0\.35 \(95% CI 0\.28\-0\.41\) to 0\.41 \(95% CI 0\.33\-0\.48\) for uncertainty span extraction, and from 0\.76 \(95% CI 0\.50\-1\.00\) to 0\.84 \(95% CI 0\.63\-1\.00\) for unknown-fact extraction. Across span tasks, LangExtract generally had lower point estimates than the corresponding highest DSPy/GEPA configuration. Among matched span predictions, mean intersection over union exceeded 0.8 for the highest-performing GPT-5.2 span-extraction configurations. @fig-results-overview summarises matched within-model comparisons across tasks for the 3 evaluated models.


### SBAR span extraction {#sec-sbar-optimized}

Among SBAR configurations, the highest overall score was achieved by DSPy/GEPA-optimised GPT-5.2 (micro-F1 0\.76 \(95% CI 0\.72\-0\.79\)), followed by DSPy/GEPA-optimised GPT-5-nano (micro-F1 0\.69 \(95% CI 0\.66\-0\.72\)) and LangExtract GPT-5.2 (micro-F1 0\.59 \(95% CI 0\.56\-0\.62\)). @tbl-sbar-optimized provides label-level results for the best-performing GPT-5.2 DSPy/GEPA SBAR configuration.

Within this GPT-5.2 SBAR comparison, macro-precision increased from 0\.41 \(95% CI 0\.37\-0\.44\) to 0\.78 \(95% CI 0\.73\-0\.82\), macro-recall from 0\.69 \(95% CI 0\.63\-0\.75\) to 0\.73 \(95% CI 0\.69\-0\.78\), and macro-F1 from 0\.49 \(95% CI 0\.46\-0\.53\) to 0\.75 \(95% CI 0\.71\-0\.79\). Span-boundary agreement among matched predictions was strongest for SITUATION and RECOMMENDATION, as shown by the label-level mean IoU estimates in @tbl-sbar-optimized.

### Checklist task {#sec-checklist}

For checklist prediction, the best overall result was achieved by DSPy/GEPA-optimised GPT-5.2 (micro-F1 0\.85 \(95% CI 0\.83\-0\.88\), macro-F1 0\.73 \(95% CI 0\.63\-0\.76\), support-weighted F1 0\.85 \(95% CI 0\.82\-0\.88\)). DSPy/GEPA-optimised GPT-5-nano also performed competitively (micro-F1 0\.81 \(95% CI 0\.78\-0\.84\)), while MedGemma 27B reached micro-F1 0\.76 \(95% CI 0\.74\-0\.79\)). @tbl-checklist-grouped presents grouped per-label estimates for accuracy, precision, recall, and F1 for the best-performing GPT-5.2 checklist model. 

### Uncertainty and unknown-fact span extraction {#sec-uncertainty-results}

Broad uncertainty-span extraction included all annotated uncertainty categories (hedging, vagueness, unknown facts, indefinite timing, source uncertainty, procedural uncertainty, and responsibility uncertainty) and had the lowest micro-F1 of the evaluated tasks. The highest DSPy/GEPA point estimate used GPT\-5\.2 and achieved precision 0\.32 \(95% CI 0\.26\-0\.39\), recall 0\.56 \(95% CI 0\.44\-0\.67\), micro-F1 0\.41 \(95% CI 0\.33\-0\.48\), and mean IoU 0\.83 \(95% CI 0\.77\-0\.90\). The highest unoptimised-prompt baseline had micro-F1 0\.35 \(95% CI 0\.28\-0\.41\), and the highest LangExtract uncertainty result had micro-F1 0\.24 \(95% CI 0\.17\-0\.31\).

The narrower unknown-fact sub-task had higher point estimates. DSPy/GEPA reached micro-F1 0\.84 \(95% CI 0\.63\-1\.00\) with GPT\-5\.2, compared with the highest baseline micro-F1 of 0\.76 \(95% CI 0\.50\-1\.00\) and the LangExtract GPT-5.2 micro-F1 of 0\.56 \(95% CI 0\.29\-0\.72\). These results concern explicitly stated lack of knowledge, not facts absent from the transcript. @tbl-uncertainty-task-results summarises the uncertainty and unknown-fact span results used for these comparisons.


## Discussion {#sec-discussion}

### Principal Results

This study identified that prompt optimisation with GEPA outperformed baseline evaluations across all of the comparisons, with higher aggregate performance for checklist prediction than for SBAR span extraction. This pattern is expected because using generative AI to perform clinical natural language processing tasks is sensitive to framing, label definitions, and example selection. GEPA was explicitly supplied with task-specific scores and error feedback that could align instructions more closely with the annotation scheme [@sivarajkumar2024prompting_strategies_clinical_nlp; @agrawal2025gepa]. Optimisation was particularly useful for improving targeted span extraction across the SBAR task.

The difference in accuracy that we identified between checklist prediction and SBAR span extraction provides an important insight to consider for designing AI applications to support clinical handover. Checklist prediction is a simpler structured-output task where each item is a bounded present/absent judgement. By contrast, SBAR extraction requires the model to locate clinically relevant text, assign a communicative category, and reproduce appropriate span boundaries. Structured handover tools aim to make minimum content predictable and reduce omissions, while still requiring adaptation to local clinical workflow [@haig2006sbar_shared_mental_model; @riesenberg2010nursing_handoffs; @bukoh2020structured_handover_review]. As such, bounded present/absent outputs may be useful when the intended function is to prompt review of missing elements or support audit and quality monitoring. It is fortunate then that the task with strongest performance in our study was those most closely aligned with a recurring mechanism of handover-related harm, namely omitted or incomplete information [@ong2011handoff_failures; @manser2011effective_handover; @riesenberg2010nursing_handoffs]. Although checklist prediction achieved high accuracy, the consequences of both false positives and false negatives should be considered. A false negative would usually create review burden by prompting clarification of an item that was already communicated, but a false positive could create more serious false reassurance that a safety-critical element was conveyed when it was not. In addition, a checklist item not detected in the captured transcript is not equivalent to clinically necessary information having been omitted. A safer near-term design could therefore present AI outputs as source-linked prompts for clinician review, with clear distinctions between detected in transcript, not detected, and requires verification. Evidence has indicated that effective clinical decision support is most useful when integrated into a workflow as actionable and readily available support at the time and place of decision-making, while also recognising the risk of automation bias when clinicians over-rely on system outputs [@kawamoto2005clinical_decision_support; @goddard2012automation_bias; @challen2019ai_bias_safety].

Registered nurse annotation agreement was comparatively high for checklist and SBAR annotations but very low for the broad uncertainty taxonomy. The latter finding indicates that vague, hedged, or context-dependent communication was difficult to operationalise consistently even with written guidance. However, the combination of low kappa and relatively high matched IoU scores for broad uncertainty indicates that the principal difficulty was deciding whether uncertainty was present, rather than locating its boundaries once identified. As all model optimisation and evaluation used the final consensus ratings between annotators, the results should therefore be interpreted as being conservative.

This study did not compare LLMs with non-LLM-based methods of structured information extraction. It therefore provides evidence about relative prompt optimisation approaches and model configurations, not the superiority of LLMs over conventional information extraction. Nevertheless, evaluating LLM-based approaches is a logical next step in this handover-specific research programme. In the original NICTA benchmark, a feature-engineered conditional random field achieved macro-F1 0.702 across 35 handover-form categories, but performance was markedly uneven. F1 was 0.217 for the more abstract “other observations” category and 0.496 for future-care goals, tasks, and expected outcomes, and the error analysis identified clinically relevant missed and misclassified information [@suominen2015benchmarking]. These limitations were concentrated in categories requiring greater contextual differentiation. Instruction-tuned LLMs therefore warranted evaluation as a potentially more flexible approach to context-dependent handover information, particularly where prompt optimisation can adapt extraction behaviour using a modest labelled set.

The optimised SBAR extraction model achieved high span-boundary overlap among matched spans in our study. It should be considered, though, that SBAR is an idealised communication framework, while real clinical handovers often move non-linearly, revisit information, distribute relevant details across the conversation, or place content between categories. For implementation, this suggests that extracted spans could be displayed with a link to their source transcript context and should support clinician review of what was said, rather than automatically transforming the handover into a definitive structured note. This preserves the benefits of structured communication while avoiding over-compression of clinically meaningful narrative context [@haig2006sbar_shared_mental_model; @rosenbloom2011structured_flexible_documentation; @cohen2010handoffs_literature_review].


### Implementation Considerations

The appropriate balance between recall and precision may differ by use case when considering implementation of the tasks we evaluated in this study into an AI tool to support clinical handover. For real-time clinician support, the intervention should be framed as shaping safe handover dialogue rather than only producing a structured output after the exchange has ended. Higher recall may be appropriate where the system displays non-detected checklist items under an “Items to check” heading and allows the receiving clinician to mark each item as addressed, not relevant, or requiring clarification. This presentation would avoid implying that the information was definitely omitted while supporting clarification before responsibility transfers. For retrospective audit or compliance monitoring, precision becomes more important because false positives could overestimate handover quality and obscure residual safety risks. The present checklist results, with high recall but non-trivial false positives in several categories, therefore support cautious separation of two implementation pathways: real-time clinician-facing gap prompts, and separately validated audit/reporting workflows [@redley2018handover_audit_tool; @manser2011effective_handover; @kawamoto2005clinical_decision_support].

Gold standard handovers include active verification and shared situational awareness rather than passive transfer of uncertain information [@patterson2004handoff_strategies; @cohen2010handoffs_literature_review; @leonard2004human_factor; @starmer2014handoff_program]. Evaluation results of the broad uncertainty-span performance indicate that even a GEPA optimised prompt with a state-of-the-art proprietary large language model is not yet reliable for detecting the full range of ambiguous, hedged, or context-dependent communication. This should be interpreted against the clinical reality that uncertainty is also difficult for humans to recognise and act on consistently, particularly under time pressure or when concern is communicated indirectly. Many forms of clinically important uncertainty are implicit and embedded in shared team understanding: phrases such as "he seems off today" or "not quite themself" can convey concern through context, trajectory, and prior knowledge rather than through explicit wording that a model can reliably extract. By contrast, the comparatively more accurate unknown-fact extraction results suggest a potentially useful role for identifying explicit information gaps that can be converted into clarification prompts for the receiving clinician. As foundation models continue to improve, accuracy on this task may also improve, although reliable use in safety-critical handover settings will still require ongoing empirical validation.

Relatedly, it is highly likely that further gains could be realised after implementation independent of general improvements in underlying large language model capability. If clinicians review AI-generated handover outputs and corrections are retained as labelled examples, subsequent optimisation cycles using the same GEPA prompt optimisation pipeline we used in this study would plausibly improve performance over time. Prior clinical natural language processing active-learning studies have demonstrated that selectively labelled examples can improve data efficiency for both clinical text classification and clinical named-entity recognition [@figueroa2012active_learning_clinical_text; @chen2015active_learning_ner_clinical_text]. In practice, any such learning cycle should be treated as a controlled quality-improvement process, with monitoring and re-evaluation before revised prompts or models are released into clinical use [@feng2022clinical_ai_quality_improvement].

Finally, it should be noted that this study measured extraction and classification performance against annotated labels, but did not evaluate whether AI outputs changed clinician questioning, closed-loop communication, task completion, escalation, near-miss detection, or patient outcomes. Technical metrics are necessary but insufficient for determining whether a clinical AI tool improves safety in practice [@kelly2019clinical_ai_impact; @challen2019ai_bias_safety; @starmer2014handoff_program]. Recent commentary and discursive work on nursing automation similarly argue that evaluation should move beyond time saved to examine how AI redistributes nursing work toward review and verification, whether tools meet end-user-defined utility thresholds, whether performance is equitable across linguistic and workforce groups, and how automation can be integrated without undermining professional values or relational care [@ronquillo2026beyond_time_saved; @pepito2025automation_nursing]. Even accurate prompts may create new risks if clinicians over-trust them, ignore non-highlighted issues, experience alert fatigue, or redirect attention away from direct patient/carer engagement. Human factors testing should therefore assess reliance, trust calibration, interruption burden, and the usability of source-linked evidence before deployment [@goddard2012automation_bias; @leonard2004human_factor; @sittig2010sociotechnical_model].


### Limitations {#sec-limitations}

The synthetic conversational handovers used in this study may not fully represent those performed in actual clinical practice where interruptions, time pressure, environmental noise, non-verbal cues, and local team dynamics can affect what is communicated and how it is interpreted. External validation on real handover audio/transcripts from intended deployment settings is therefore required [@cohen2010handoffs_literature_review; @patterson2004handoff_strategies; @sittig2010sociotechnical_model]. External validity may be particularly limited for complex areas such as mental health, multimorbidity, and other contexts where clinical risk is often communicated through narrative nuance, formulation, behavioural change, staff concern, or tone of interaction rather than discrete data points. These contexts strengthen the need for AI outputs to remain clinician-reviewed, source-linked supports rather than autonomous clinical documentation.

It should be noted that several checklist items had low prevalence in the evaluation partition. Future evaluation should oversample or otherwise specifically test these items [@challen2019ai_bias_safety; @riesenberg2010nursing_handoffs]. Furthermore, the reference standard retained only annotations agreed by both reviewers. This creates a conservative and reproducible benchmark, but may exclude ambiguous or contested communication.

The evaluation corpus included synthetic transcripts generated in part with GPT-5. Although the uses were distinct, information extraction tasks were performed on synthesised language from a model from the same LLM provider, which is potentially susceptible to model familiarity. Synthetic transcripts may also be more predictable than authentic handover communication. In addition, the highest-performing models evaluated in this study were hosted proprietary systems that are updated and managed externally. Model updates, infrastructure variability, or configuration changes could alter performance over time in ways that are not fully transparent or controllable, with implications for reproducibility, governance, and consistency in clinical settings where reliability is critical. Locally hosted models may offer greater control over model versioning and deployment conditions, but can require substantial compute and may be less responsive to rapid improvements in frontier model capability.

Our evaluation focused on downstream extraction from transcripts and did not separately quantify transcription accuracy, speaker attribution, or the effect of noisy audio. In a real-time handover system, errors introduced before the LLM stage could alter checklist detection, SBAR span extraction, and uncertainty identification. End-to-end evaluation should therefore include audio capture, transcription, diarisation, and transcript-to-output performance under realistic clinical conditions [@sittig2010sociotechnical_model; @ong2011handoff_failures]. We used a checklist that was developed at a local institution based on local priorities for communication during clinical nursing handover. Other wards, specialties, transfer types, or jurisdictions may prioritise different minimum datasets or use different terminology. Implementation would therefore require local mapping of labels to existing handover policy, audit tools, escalation pathways, and documentation workflows, followed by local validation rather than direct transfer of the reported performance estimates [@redley2018handover_audit_tool; @riesenberg2010nursing_handoffs; @sittig2010sociotechnical_model].

### Conclusions {#sec-conclusion}

In this evaluation of AI-assisted clinical handover tasks, prompt optimisation consistently improved matched model performance. Accuracy was highest for bounded checklist prediction and SBAR span extraction. Broad uncertainty detection remained insufficiently reliable, while explicit unknown-fact extraction suggests a narrower but important role for prompting clarification when missing knowledge is directly expressed. Before clinical deployment, these approaches require external validation on real handover audio and transcripts, end-to-end testing of transcription and diarisation effects, local calibration of checklist definitions, human-factors evaluation of trust, reliance, workflow burden, and downstream safety outcomes, and structured assessment of organisational readiness for AI implementation in nursing care [@seibert2026aincra].

## Acknowledgments {#sec-acknowledgments}

The authors used OpenAI ChatGPT/Codex during manuscript preparation to assist with editing, formatting, and generation of tables and figures to present results. The authors reviewed, verified, and approved all analytic decisions, references, and final manuscript text. 

## Funding {#sec-funding}

This work was supported by The Prince Charles Hospital Foundation Collaboration Grant. The funder had no role in study design, data collection, analysis, interpretation, manuscript preparation, and the decision to submit.

## Conflicts of Interest {#sec-conflicts}

The authors declare no conflicts of interest.

## Data Availability {#sec-data-availability}

The synthetic handover transcripts, annotation schema, evaluation outputs, and analysis code are available in a public repository on Figshare at [https://doi.org/10.6084/m9.figshare.32658084](https://doi.org/10.6084/m9.figshare.32658084). The NICTA Synthetic Nursing Handover Dataset is publicly available from its original source.

## Authors' Contributions {#sec-author-contributions}

Conceptualization: AC, AT. Data curation: AC. Formal analysis: AC. Investigation: AC, AH, JS. Methodology: AC, AH, AT, DL, KD. Project administration: AC. Software: AC. Visualization: AC. Writing - original draft: AC. Writing - review and editing: AC, AH, JS, GX, DL, TM, KD, AT.

## Abbreviations {#sec-abbreviations}

| Abbreviation | Definition |
| :--- | :--- |
| AI | artificial intelligence |
| CI | confidence interval |
| COI | conflict of interest |
| DSPy | framework for declarative language model programming |
| F1 | harmonic mean of precision and recall |
| FN | false negative |
| FP | false positive |
| GEPA | Genetic-Pareto prompt optimisation |
| GPT | generative pretrained transformer |
| IoU | intersection over union |
| LLM | large language model |
| NICTA | National Information and Communications Technology Australia |
| ORCID | Open Researcher and Contributor ID |
| SBAR | Situation, Background, Assessment, Recommendation |
| TN | true negative |
| TP | true positive |

{{< pagebreak >}}

# References

::: {#refs}
:::

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
| Hedge / Probability Language | The speaker indicates partial confidence or doubt about information. | “I think ENT reviewed him”; “He should be going to theatre soon.” |
| Vague / Qualitative Expression | Information is described using imprecise or subjective language. | “He looks fine now”; “Seems okay.” |
| Unknown Fact / Explicit Lack of Knowledge | The speaker openly states missing knowledge or incomplete data. | “Not sure if consent’s been signed”; “I don’t know his allergies.” |
| Indefinite Timing | Timing or schedule for an event is vague or lacks precision. | “Later today”; “After the round.” |
| Source Uncertainty | Information relies on a second-hand or unverifiable source. | “ENT said he’s on the list”; “Night nurse told me.” |
| Procedural Uncertainty | The next step in care is unclear or the plan is not explicitly stated. | “You might want to check his IV.” |
| Responsibility Uncertainty | A required task or follow-up is mentioned, but it is unclear who is responsible for performing it. | “Bloods to be checked later”; “Needs review this afternoon.” |

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


Per-label SBAR metrics for the best-performing GPT-5.2 DSPy/GEPA model
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

: Grouped per-label checklist performance for the best-performing GPT-5.2 DSPy/GEPA model {#tbl-checklist-grouped tbl-colwidths="[44,14,14,14,14]"}


{{< pagebreak >}}

*Table 5 (continued)*

| Checklist item | Accuracy | Precision | Recall | F1 |
| -------------------------------------------- | -------------: | -------------: | -------------: | -------------: |
| **Recommendation** |  |  |  |  |
| Care plan/pathway actions to follow up | 0.90 (0.80-0.98) | 0.90 (0.80-0.98) | 1.00 | 0.95 (0.89-0.99) |
| Asked patient/carer about goals and preferences | 0.80 (0.67-0.90) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) |
| Discharge plan | 0.98 (0.94-1.00) | 0.80 (0.33-1.00) | 1.00 | 0.89 (0.50-1.00) |
| Critical actions required | 0.92 (0.84-0.98) | 0.20 (0.00-0.67) | 1.00 | 0.33 (0.00-0.80) |
| **Patient Involvement** |  |  |  |  |
| Introduction of clinicians involved in handover to patient/carer | 0.84 (0.73-0.94) | 0.71 (0.55-0.87) | 1.00 | 0.83 (0.71-0.93) |
| Invitation for patient/carer to participate in handover | 0.82 (0.69-0.92) | 0.65 (0.44-0.83) | 0.94 (0.78-1.00) | 0.77 (0.59-0.89) |

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


Uncertainty and unknown-fact span extraction
:::

*Legend:* IoU = intersection over union for matched span boundaries; F1 = harmonic mean of precision and recall.

:::::
