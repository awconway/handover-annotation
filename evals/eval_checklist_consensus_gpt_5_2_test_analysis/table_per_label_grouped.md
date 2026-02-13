| Category | Label | Support | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identification | ID check of 3 patient identifiers | 1 | 1 | 0 | 0 | 48 | 1.000 | 1.000 | 1.000 |
| Situation | Primary diagnosis \| reason for admission | 45 | 45 | 2 | 0 | 2 | 0.957 | 1.000 | 0.978 |
| Situation | Current status (awaiting tests/procedures, on interim orders/plan) | 31 | 22 | 8 | 9 | 10 | 0.733 | 0.710 | 0.721 |
| Situation | Significant events or complications | 10 | 5 | 3 | 5 | 36 | 0.625 | 0.500 | 0.556 |
| Situation | **Subtotal (Situation)** | **86** | **72** | **13** | **14** | **48** | **0.847** | **0.837** | **0.842** |
| Background | Alerts - allergies | 18 | 16 | 2 | 2 | 29 | 0.889 | 0.889 | 0.889 |
| Background | Relevant clinical and social history \| comorbidities | 17 | 17 | 1 | 0 | 31 | 0.944 | 1.000 | 0.971 |
| Background | Alerts - falls risk | 3 | 2 | 0 | 1 | 46 | 1.000 | 0.667 | 0.800 |
| Background | Alerts - pressure injury risk | 3 | 2 | 0 | 1 | 46 | 1.000 | 0.667 | 0.800 |
| Background | Advanced care planning | 1 | 1 | 0 | 0 | 48 | 1.000 | 1.000 | 1.000 |
| Background | **Subtotal (Background)** | **42** | **38** | **3** | **4** | **200** | **0.927** | **0.905** | **0.916** |
| Assessment | Observations \| Q-ADDS \| recent escalations | 40 | 38 | 2 | 2 | 7 | 0.950 | 0.950 | 0.950 |
| Assessment | Medication chart \| flag high risk meds | 25 | 24 | 4 | 1 | 20 | 0.857 | 0.960 | 0.906 |
| Assessment | Devices \| lines \| vascular access | 23 | 23 | 3 | 0 | 23 | 0.885 | 1.000 | 0.939 |
| Assessment | Mobility \| aids | 17 | 16 | 4 | 1 | 28 | 0.800 | 0.941 | 0.865 |
| Assessment | Pain management | 17 | 16 | 2 | 1 | 30 | 0.889 | 0.941 | 0.914 |
| Assessment | Infusions | 15 | 8 | 1 | 7 | 33 | 0.889 | 0.533 | 0.667 |
| Assessment | Pathology | 15 | 14 | 5 | 1 | 29 | 0.737 | 0.933 | 0.824 |
| Assessment | Nutrition \| restrictions | 14 | 14 | 6 | 0 | 29 | 0.700 | 1.000 | 0.824 |
| Assessment | Fluid balance \| restrictions | 9 | 7 | 3 | 2 | 37 | 0.700 | 0.778 | 0.737 |
| Assessment | Skin integrity \| interventions | 6 | 6 | 5 | 0 | 38 | 0.545 | 1.000 | 0.706 |
| Assessment | Critical monitoring \| alarms | 0 | 0 | 2 | 0 | 47 | 0.000 | 0.000 | 0.000 |
| Assessment | **Subtotal (Assessment)** | **181** | **166** | **37** | **15** | **321** | **0.818** | **0.917** | **0.865** |
| Recommendation | Care plan/pathway actions to follow up | 44 | 44 | 5 | 0 | 0 | 0.898 | 1.000 | 0.946 |
| Recommendation | Asked patient/carer about goals and preferences | 9 | 0 | 1 | 9 | 39 | 0.000 | 0.000 | 0.000 |
| Recommendation | Discharge plan | 4 | 4 | 1 | 0 | 44 | 0.800 | 1.000 | 0.889 |
| Recommendation | Critical actions required | 1 | 1 | 4 | 0 | 44 | 0.200 | 1.000 | 0.333 |
| Recommendation | **Subtotal (Recommendation)** | **58** | **49** | **11** | **9** | **127** | **0.817** | **0.845** | **0.831** |
| Patient Involvement | Introduction of clinicians involved in handover to patient/carer | 20 | 20 | 8 | 0 | 21 | 0.714 | 1.000 | 0.833 |
| Patient Involvement | Invitation for patient/carer to participate in handover | 16 | 15 | 8 | 1 | 25 | 0.652 | 0.938 | 0.769 |
| Patient Involvement | **Subtotal (Patient Involvement)** | **36** | **35** | **16** | **1** | **46** | **0.686** | **0.972** | **0.805** |
| **Overall (Micro)** | **All labels pooled** | **404** | **361** | **80** | **43** | **790** | **0.819** | **0.894** | **0.854** |
| **Overall (Macro)** | **Unweighted label mean** | **-** | **-** | **-** | **-** | **-** | **0.745** | **0.823** | **0.762** |
| **Overall (Support-Weighted)** | **Weighted by label support** | **404** | **-** | **-** | **-** | **-** | **0.822** | **0.894** | **0.849** |
