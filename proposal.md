# Co-designing an Artificial intelligence Tool for Clinical Handovers (the CATCH collaborative)

**Research Area.** Clinical handovers are fundamental communication
processes in healthcare, involving the transfer of critical patient
information and professional responsibility between clinicians or teams.
These transitions occur frequently, such as during shift changes,
patient transfers between units or facilities, and discharge back to
community care^1^, and can be complex. Handovers are recognised as
high-risk periods susceptible to communication errors. Ineffective
communication at transitions of care is a significant contributor to
preventable adverse events.^2^ Poor handover can lead to delayed
diagnosis and treatment, medication errors, unnecessary tests, and
preventable readmissions, wasting resources and causing patient harm.^3^
Factors contributing to ineffective handovers include time pressures,
lack of standardised procedures, incomplete or inaccurate information
transfer, distractions, and communication breakdowns.^3^

**Clinical Need.** There is a persistent clinical need to enhance the
reliability, accuracy, completeness, and efficiency of clinical
handovers, which is recognised by the National Safety and Quality Health
Service (NSQHS) Standards *Communicating for Safety Standard*. This
standard specifically highlights clinical handover as a high-risk time
requiring structured processes to ensure critical information is
accurately transferred and responsibility is clearly assigned. Despite
these standards and the use of tools to promote structured communication
of information (such as the ISBAR format), achieving consistently
effective handover communication, especially in complex or high-pressure
situations, remains a significant challenge.^4^ Clinicians require
support to consistently meet these needs and adhere to these standards,
minimise errors, and ensure critical information is communicated
effectively, particularly during complex transitions or for patients
with multifaceted needs (e.g., mental health). Addressing these
persistent handover challenges requires innovative solutions that can
augment clinician capabilities in real time. Generative AI offers a
powerful opportunity to support clinicians by rapidly processing
information, identifying key details, and generating context-specific
recommendations that support optimal communication of information at
handover.

  -----------------------------------------------------------------------
  ![](media/image4.svg){width="0.17716535433070865in"
  height="0.17716535433070865in"}**Aim**
  -----------------------------------------------------------------------
  The aim of this project is to *rapidly* co-design a digital healthcare
  application that uses generative AI to provide support and guidance for
  clinical handovers. A co-design approach will guide the design, but we
  envision the system to possess the core functional capabilities to
  audio-record, transcribe and assess clinical handovers using a large
  language model before communicating AI-generated recommendations.

  -----------------------------------------------------------------------

# Research Design

## Phase 1 objective: Create a generative AI pipeline capable of producing accurate recommendations for communicating critical information during handover. 

*This phase will distil expert clinical judgment capabilities into the
generative AI large language model by annotating datasets representative
of the type of complex handovers encountered in clinical practice across
specialties followed by a systematic process of prompt optimisation and
fine-tuning.*

*Data sources:* 1) *The publicly available NICTA Synthetic Nursing
Handover Dataset*.^5^ This dataset includes 300 realistic recordings of
clinical handovers made by a Registered Nurse (RN) based on patient
profiles with cardiovascular, neurological, renal and respiratory
conditions. Handover monologues recorded by the RN were created from
comprehensive patient profiles that included information on the
patient's name, age, admission story, in-patient time, and familiarity
to the nurses giving and receiving the handover. The RN was encouraged
to envision herself working in a medical ward, providing verbal
handovers to another nurse at the patient\'s bedside during a shift
change.^5^ 2) *A dataset comprising 100 transcribed recordings of a
junior doctor reporting to a senior doctor based on documents from real
clinical cases* related to either a medical patient with respiratory
failure or a patient with acute abdominal injury requiring urgent
surgery.^6^ The dataset comprises a total of 1895 sentences of English
language text, which was produced at the Queen Elizabeth Hospital in
Hong Kong. 3) *A 'local TPCH' synthetic dataset that we will develop
comprising handovers depicting complex clinical handovers used in mental
health settings and other high-risk contexts* (e.g. populations with
complex care needs including older adults, patients with delirium and
those transitioning between acute and community care settings)*.* The
same processes used to create the NICTA dataset will be applied, with
handovers created based on comprehensive and realistic profiles to be
supported by a process of data augmentation.^7^

*Annotation:* Two research assistants with clinical experience related
to the type of handover will independently annotate handover transcripts
using a structured evaluation rubric to assess clarity, accuracy,
completeness, adherence to local (TPCH) and national recommendations
(NSQHS), and the presence of critical safety-related information. A
random sample of 100 transcripts from dataset 1, 20 transcripts from
dataset 2 and a further 80 'local' transcripts will be annotated
(n=200). Annotators will provide explicit recommendations for improving
the handover dialogue by emphasising critical information additions or
removals, reordering of information to enhance logical flow, and
modifications to clinical terminology accuracy and consistency.
Annotators will be instructed about how to articulate their reasoning
behind each recommended change to ensure their feedback can be directly
used for shaping prompt optimisations or fine-tuning the generative
model. Inter-rater reliability will be monitored and annotation
consensus meetings with the whole investigator team and consumer partner
will be held to resolve discrepancies and ensure annotation consistency
and quality.

*Analysis:* The python package DSPy will be used for iterative
prototyping and optimising generative AI prompt pipelines. ^8^ Model
fine-tuning, again facilitate by DSPy will be considered if initial
prompt optimisation alone proves insufficient. Large language models to
be evaluated include smaller open-source generative models (e.g.,
LLaMA^9^ and Mistral^10^ variants between 7B-13B parameters) due to
their proven efficiency and feasibility for targeted fine-tuning within
research computing resource constraints. The larger proprietary models
available through QLD Health Azure cloud service accounts (e.g., GPT-4)
will also be assessed for baseline comparison of performance. Annotated
handover data will form the gold standard reference for rigorously
evaluating both prompt refinements and any targeted model fine-tuning
guided by DSPy's optimisation algorithms. Performance metrics and
human-AI annotation agreement, will inform these decisions, ensuring the
AI system reliably identifies key clinical handover components and
provides accurate recommendations.

## Phase 2 objective: identify how a clinician will interact with the application during a clinical handover.

This phase engages consumers and end-users (nurses, physicians and
allied health clinicians) in the design of the AI application. It will
proceed over three steps: (1) consumer and clinician semi-structured
interviews; (2) co-creation workshops; and 3) usability testing.

*Step 1:* Qualitative interviews will be conducted with at least 5
clinicians and consumers to explore perspectives about perceived needs,
priorities and preferences for receiving evaluations and feedback on
communication during clinical handovers, paying particular attention to
how these needs vary across different handover contexts (e.g.,
shift-to-shift within a ward, inter-departmental, hospital-to-community
transitions). A key objective of this step is to identify specific
handover scenarios where the AI application could provide value and
understand the necessary flexibility required to accommodate diverse
workflows and challenges, such as managing information transfer across
care boundaries. Semi-structured interviews will be conducted based on
methods that will facilitate rapid data collection and analysis. Design
thinking tools like empathy maps for consumers and "I like, I wish, what
if" questioning for clinician end-users will be used to probe these
contextual differences. Data generated from these tools will be used to
develop design artifacts, such as user journey maps reflecting different
handover pathways and a stakeholder needs table that accounts for
varying contexts.

*Step 2:* Two interactive co-creation workshops involving at least 10
consumers and clinicians will then be conducted with the focus on
translating needs and preferences identified in step 1 into a
low-fidelity prototype design of the user interface. User experience
(UX) design methods will be applied during the workshops to explore
problems that need solving, brainstorm ideas, generate possible
solutions, and make decisions together about specific design elements
and user interactions.^11^ The workshops will result in the production
of a paper wireframe representation of the application, which is a
static, low-fidelity schematic made up of several visual components that
will communicate to the software developer how the application should be
built.^12^

+-----------------------------------------------------------------------+
| #### *                                                                |
| *Security considerations for application development and deployment** |
+=======================================================================+
| Based on these workshops, a high-fidelity prototype of the            |
| application will be developed by a software developer using the       |
| **Microsoft Power Apps platform**. This platform ensures that access  |
| to the application is secure and authenticated through Queensland     |
| Health accounts in accordance with current guidance and               |
| recommendations from Digital Metro North for developing generative AI |
| applications.                                                         |
+-----------------------------------------------------------------------+

*Step 3:* Usability testing of the high-fidelity application prototype
will be undertaken by at least 12 nursing, medical and allied health
clinicians from TPCH with one consumer joining each session. Recruitment
strategies will target clinicians of different ages and years of
clinical experience as well as from specific disciplines (nurses, nurse
practitioners, physicians, allied health, pharmacists) and different
service delivery settings (acute care, outpatient clinic, community) to
avoid over-representation by any specific group. Prior to user testing,
participants from steps 1 and 2 will be asked to review the prototype to
ensure it aligns with their expectations. We anticipate that 3-4
iterative testing cycles with 2-4 participants may be required to refine
the design. In each round of usability testing, clinician participants
will be asked to access and use the application to perform a clinical
handover that is scripted to suit the clinician's qualifications and
clinical experience. Participants will be instructed to "think aloud''
by expressing their thoughts, feelings, and opinions while interacting
with the prototype application.^13^ At least one consumer will be paired
with the clinician participants for each round and will be asked to
share their perspectives during testing on matters that may impact the
clinician-patient encounter. Semi-structured interview questions will
follow the think-aloud activity, focusing on generating perspectives
about ease of use, ease of understanding, acceptability and potential
impact that use of the application may have on clinician/patient
interactions. Refinements to the application based on insights from the
content analysis will be addressed before another testing cycle is
initiated.

## Phase 3 objective: evaluate the accuracy and perceived usefulness of the handover AI application

A quantitative descriptive simulation study will be conducted.
Participants will be asked to use the high-fidelity handover AI
application prototype to assist them with performing handover for a
simulated scenario. Using simulation allows for robust evaluation of the
application\'s core functionality and intended end-user perceptions in a
controlled environment in acknowledgement of that fact that the digital
infrastructure and data governance frameworks for integration of AI
tools into live clinical workflows is still an evolving space.

*Participants:* Physicians, nurses and allied health professionals will
be invited to participate. We will advertise for potential participants
through routine communication channels across the nursing and mental
health departments. The recruitment target is 30 participants over a
4-month data collection period.

*Data collection:* Data collection will take place in a private room in
the TPCH Education Centre. Participants will be partnered into dyads and
given details about a fictitious patient and clinical scenario including
all information required to perform the clinical handover, which will be
developed collaboratively by the investigators and consumer
representative on the research steering committee. In addition to the
core quantitative measurements outlined below, participants will have an
option to add open-ended comments.

-   *Duration of engagement with the AI tool:* AI handover application
    is \'in focus\' on the user\'s device.

-   *AI-generated recommendation summary:* Number, length and topics of
    generated recommendations.

-   *Perceived impact on handover time:* Measured in minutes.

-   *Ease of use:* Likert scale (\'extremely easy\' to \'extremely
    difficult\').

-   *Usefulness of AI output:* Likert scale (\'not useful at all\' to
    \'extremely useful\').

-   *AI tool recommendations used to assist task completion:* \'Yes\' or
    \'No\'.

-   *Gender bias in AI output:* Likert scale (\'not at all\' to
    \'extremely biased\').

-   *Age bias in AI output:* Likert scale (\'not at all\' to \'extremely
    biased\').

-   *Racial bias in AI output:* Likert scale (\'not at all\' to
    \'extremely biased\').

-   *Social bias in AI output:* Likert scale (\'not at all\' to
    \'extremely biased\').

-   *Disagreement by clinician with recommendations generated by the AI
    tool:* \'Yes\' or \'No\'.

-   *Perceived severity of consequences arising from accepting AI
    recommendations:* (\'trivial\' to 'severe').

-   *Ethical acceptability of using the AI recommendations*: Likert
    scale (\'not at all\' to 'entirely\').

*Data analysis:* Data will be analysed using descriptive statistics and
reported with 95% confidence intervals. Exploratory analyses will be
conducted to determine if outcomes are associated with the type of
clinical handover (based on discipline \[e.g. nurse-nurse, nurse-allied
health\], specialty \[e.g. mental health, ED\] and transition
\[admission, shift change, discharge\]). We will create visualisations
that display outcomes (ease of use, usefulness, time impact, etc)
grouped by different types of clinical handovers. Inferential statistics
will not be used to measure the strength of associations due to the
small sample size. Open-ended comments from participants will be
summarised and synthesised using content analysis.

**Potential outcomes**. This project will produce a co-designed and
usability-tested generative AI application specifically tailored to
support clinical handovers, along with a low-risk evaluation of its
performance in realistic simulated scenarios. Findings will provide
evidence about the perceived usability, usefulness, impact on workflow,
error rates, and potential biases (gender, age, racial, social) as
perceived by clinicians. As a consequence of this evaluation, this
project will yield significant insights into the ethical and legal
implications of deploying generative AI in this clinical context. These
insights will be vital for informing responsible implementation
strategies for similar AI tools across our collaborating departments at
TPCH. Furthermore, by developing a tool intended for use across
different professional groups (nursing, medical, allied health) and
different clinical settings, this project has the potential to help
overcome siloed communication patterns. An AI support tool could promote
a more consistent approach to handover communication, facilitating
better understanding and collaboration between diverse clinicians
involved in a patient\'s care. Ultimately, the project aims to provide
evidence on whether a co-designed AI application can contribute to
safer, more efficient, and collaborative patient care.

**Plan for implementation****.** Implementation will be approached
cautiously and is contingent upon results of the simulation study. If
Phase 3 results are favourable, we would proceed towards a highly
controlled, small-scale pilot implementation. Implementation efforts
will involve expanding our collaborations with AI and digital health
experts to optimise and scale the solution effectively.

## References {#references .EndNote-Bibliography-Title}

1 Johnston, K., Cassimatis, J. & Hattingh, L. Effects of inadequate
hospital clinical handover on metropolitan general practitioners in
Queensland: a qualitative study. *Australian Journal of General
Practice* **53**, 583-588 (2024).

2 Hada, A. & Coyer, F. Shift‐to‐shift nursing handover interventions
associated with improved inpatient outcomes---Falls, pressure injuries
and medication administration errors: An integrative review. *Nursing &
Health Sciences* **23**, 337-351 (2021).

3 Desmedt, M., Ulenaers, D., Grosemans, J., Hellings, J. & Bergs, J.
Clinical handover and handoff in healthcare: a systematic review of
systematic reviews. *International Journal for Quality in Health Care*
**33**, mzaa170 (2021).

4 Hada, A., Jones, L. V., Jack, L. C. & Coyer, F. Translating
evidence‐based nursing clinical handover practice in an acute care
setting: A quasi‐experimental study. *Nursing & Health Sciences* **23**,
466-476 (2021).

5 Suominen, H., Zhou, L., Hanlen, L. & Ferraro, G. Benchmarking Clinical
Speech Recognition and Information Extraction: New Data, Methods, and
Evaluations. *JMIR Med Inform* **3**, e19 (2015).
<https://doi.org/10.2196/medinform.4321>

6 Zhang, X. *et al.* in *2022 7th International Conference on
Computational Intelligence and Applications (ICCIA).* 11-16 (IEEE).

7 Zhang, X. *et al.* in *2022 IEEE International Conference on Teaching,
Assessment and Learning for Engineering (TALE).* 396-403 (IEEE).

8 Khattab, O. *et al.* Dspy: Compiling declarative language model calls
into self-improving pipelines. *arXiv preprint arXiv:2310.03714* (2023).

9 Touvron, H. *et al.* Llama: Open and efficient foundation language
models. *arXiv preprint arXiv:2302.13971* (2023).

10 Jiang, F. *Identifying and mitigating vulnerabilities in
llm-integrated applications*, University of Washington, (2024).

11 Will, T. *User experience (UX): process and methodology*,
\<<https://uiuxtrend.com/user-experience-ux-process/>\> (2020).

12 Gudoniene, D., Staneviciene, E., Buksnaitis, V. & Daley, N. The
scenarios of artificial intelligence and wireframes implementation in
engineering education. *Sustainability* **15**, 6850 (2023).

13 Woodward, M. *et al.* How to co-design a prototype of a clinical
practice tool: a framework with practical guidance and a case study.
*BMJ Quality & Safety* **33**, 258-270 (2024).
