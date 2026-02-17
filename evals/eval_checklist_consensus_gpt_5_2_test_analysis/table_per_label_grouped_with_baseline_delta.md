| Category | Label | Support | TP | FP | FN | TN | Precision | ΔPrecision vs Baseline | Recall | ΔRecall vs Baseline | F1 | ΔF1 vs Baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identification | ID check of 3 patient identifiers | 1 | 1 | 0 | 0 | 48 | 1.000 | +0.944 | 1.000 | +0.000 | 1.000 | +0.895 |
| Situation | Primary diagnosis \| reason for admission | 45 | 45 | 2 | 0 | 2 | 0.957 | +0.020 | 1.000 | +0.000 | 0.978 | +0.011 |
| Situation | Current status (awaiting tests/procedures, on interim orders/plan) | 31 | 22 | 8 | 9 | 10 | 0.733 | +0.033 | 0.710 | -0.194 | 0.721 | -0.067 |
| Situation | Significant events or complications | 10 | 5 | 3 | 5 | 36 | 0.625 | +0.292 | 0.500 | -0.100 | 0.556 | +0.127 |
| Situation | **Subtotal (Situation)** | **86** | **72** | **13** | **14** | **48** | **0.847** | **+0.102** | **0.837** | **-0.081** | **0.842** | **+0.019** |
| Background | Alerts - allergies | 18 | 16 | 2 | 2 | 29 | 0.889 | -0.011 | 0.889 | -0.111 | 0.889 | -0.058 |
| Background | Relevant clinical and social history \| comorbidities | 17 | 17 | 1 | 0 | 31 | 0.944 | +0.135 | 1.000 | +0.000 | 0.971 | +0.077 |
| Background | Alerts - falls risk | 3 | 2 | 0 | 1 | 46 | 1.000 | +0.250 | 0.667 | -0.333 | 0.800 | -0.057 |
| Background | Alerts - pressure injury risk | 3 | 2 | 0 | 1 | 46 | 1.000 | +0.000 | 0.667 | -0.333 | 0.800 | -0.200 |
| Background | Advanced care planning | 1 | 1 | 0 | 0 | 48 | 1.000 | +0.000 | 1.000 | +0.000 | 1.000 | +0.000 |
| Background | **Subtotal (Background)** | **42** | **38** | **3** | **4** | **200** | **0.927** | **+0.070** | **0.905** | **-0.095** | **0.916** | **-0.007** |
| Assessment | Observations \| Q-ADDS \| recent escalations | 40 | 38 | 2 | 2 | 7 | 0.950 | +0.023 | 0.950 | +0.000 | 0.950 | +0.012 |
| Assessment | Medication chart \| flag high risk meds | 25 | 24 | 4 | 1 | 20 | 0.857 | +0.030 | 0.960 | +0.000 | 0.906 | +0.017 |
| Assessment | Devices \| lines \| vascular access | 23 | 23 | 3 | 0 | 23 | 0.885 | +0.000 | 1.000 | +0.000 | 0.939 | +0.000 |
| Assessment | Mobility \| aids | 17 | 16 | 4 | 1 | 28 | 0.800 | +0.027 | 0.941 | -0.059 | 0.865 | -0.007 |
| Assessment | Pain management | 17 | 16 | 2 | 1 | 30 | 0.889 | +0.222 | 0.941 | +0.000 | 0.914 | +0.134 |
| Assessment | Infusions | 15 | 8 | 1 | 7 | 33 | 0.889 | +0.089 | 0.533 | -0.267 | 0.667 | -0.133 |
| Assessment | Pathology | 15 | 14 | 5 | 1 | 29 | 0.737 | -0.063 | 0.933 | +0.133 | 0.824 | +0.024 |
| Assessment | Nutrition \| restrictions | 14 | 14 | 6 | 0 | 29 | 0.700 | +0.000 | 1.000 | +0.000 | 0.824 | +0.000 |
| Assessment | Fluid balance \| restrictions | 9 | 7 | 3 | 2 | 37 | 0.700 | -0.050 | 0.778 | -0.222 | 0.737 | -0.120 |
| Assessment | Skin integrity \| interventions | 6 | 6 | 5 | 0 | 38 | 0.545 | +0.045 | 1.000 | +0.000 | 0.706 | +0.039 |
| Assessment | Critical monitoring \| alarms | 0 | 0 | 2 | 0 | 47 | 0.000 | +0.000 | 0.000 | +0.000 | 0.000 | +0.000 |
| Assessment | **Subtotal (Assessment)** | **181** | **166** | **37** | **15** | **321** | **0.818** | **+0.044** | **0.917** | **-0.028** | **0.865** | **+0.014** |
| Recommendation | Care plan/pathway actions to follow up | 44 | 44 | 5 | 0 | 0 | 0.898 | -0.035 | 1.000 | +0.045 | 0.946 | +0.002 |
| Recommendation | Asked patient/carer about goals and preferences | 9 | 0 | 1 | 9 | 39 | 0.000 | -0.500 | 0.000 | -0.222 | 0.000 | -0.308 |
| Recommendation | Discharge plan | 4 | 4 | 1 | 0 | 44 | 0.800 | +0.000 | 1.000 | +0.000 | 0.889 | +0.000 |
| Recommendation | Critical actions required | 1 | 1 | 4 | 0 | 44 | 0.200 | +0.170 | 1.000 | +0.000 | 0.333 | +0.275 |
| Recommendation | **Subtotal (Recommendation)** | **58** | **49** | **11** | **9** | **127** | **0.817** | **+0.253** | **0.845** | **+0.000** | **0.831** | **+0.155** |
| Patient Involvement | Introduction of clinicians involved in handover to patient/carer | 20 | 20 | 8 | 0 | 21 | 0.714 | -0.036 | 1.000 | +0.850 | 0.833 | +0.583 |
| Patient Involvement | Invitation for patient/carer to participate in handover | 16 | 15 | 8 | 1 | 25 | 0.652 | -0.014 | 0.938 | -0.062 | 0.769 | -0.031 |
| Patient Involvement | **Subtotal (Patient Involvement)** | **36** | **35** | **16** | **1** | **46** | **0.686** | **+0.290** | **0.972** | **+0.444** | **0.805** | **+0.352** |
| **Overall (Micro)** | **All labels pooled** | **404** | **361** | **80** | **43** | **790** | **0.819** | **+0.136** | **0.894** | **+0.000** | **0.854** | **+0.081** |
| **Overall (Macro)** | **Unweighted label mean** | **-** | **-** | **-** | **-** | **-** | **0.745** | **+0.086** | **0.823** | **-0.002** | **0.762** | **+0.073** |
| **Overall (Support-Weighted)** | **Weighted by label support** | **404** | **-** | **-** | **-** | **-** | **0.822** | **+0.020** | **0.894** | **+0.000** | **0.849** | **+0.023** |
