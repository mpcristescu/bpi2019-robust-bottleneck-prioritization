# BPI Challenge 2019 : Phase 1 Feasibility Report

## 1. Dataset audit

- Traces/cases: **251,734**
- Events: **1,595,923**
- Activities: **42**
- Observed resources: **628**
- Resource coverage: **100.00%**
- Variants: **11,973**
- Time range: **1948-01-26T22:59:00+00:00 → 2020-04-09T21:59:00+00:00**
- Cases with timestamp-order violations: **0**

## 2. Lifecycle evidence

Observed lifecycle values: `{}`
Direct start/complete processing-time measurement supported: **False**

## 3. Case duration

- median: 1537.0583333333334
- P95: 3415.100833333332
- P99: 5268.056166666664

## 4. Inter-event delay reservoir sample

- gaps seen: 1,344,189
- sampled: 200,000
- median: 59.45
- P95: 1537.5858333333326

## 5. Top activities

- Record Goods Receipt: 314097
- Create Purchase Order Item: 251734
- Record Invoice Receipt: 228760
- Vendor creates invoice: 219919
- Clear Invoice: 194393
- Record Service Entry Sheet: 164975
- Remove Payment Block: 57136
- Create Purchase Requisition Item: 46592
- Receive Order Confirmation: 32065
- Change Quantity: 21449
- Change Price: 12423
- Delete Purchase Order Item: 8875
- Change Approval for Purchase Order: 7541
- Cancel Invoice Receipt: 7096
- Vendor creates debit memo: 6255

## 6. Top resources

- NONE: 399090
- user_002: 166353
- user_029: 71539
- user_020: 39770
- batch_06: 38100
- user_013: 35069
- user_001: 34563
- user_012: 32707
- user_019: 29764
- user_235: 28336

## 7. Dominant variants

- n=50286 | len=5 | Create Purchase Order Item -> Vendor creates invoice -> Record Goods Receipt -> Record Invoice Receipt -> Clear Invoice
- n=30798 | len=5 | Create Purchase Order Item -> Record Goods Receipt -> Vendor creates invoice -> Record Invoice Receipt -> Clear Invoice
- n=12214 | len=2 | Create Purchase Order Item -> Record Goods Receipt
- n=11383 | len=6 | Create Purchase Order Item -> Vendor creates invoice -> Record Goods Receipt -> Record Invoice Receipt -> Remove Payment Block -> Clear Invoice
- n=9694 | len=6 | Create Purchase Order Item -> Receive Order Confirmation -> Record Goods Receipt -> Vendor creates invoice -> Record Invoice Receipt -> Clear Invoice
- n=8921 | len=6 | Create Purchase Requisition Item -> Create Purchase Order Item -> Vendor creates invoice -> Record Goods Receipt -> Record Invoice Receipt -> Clear Invoice
- n=8835 | len=6 | Create Purchase Order Item -> Vendor creates invoice -> Record Invoice Receipt -> Record Goods Receipt -> Remove Payment Block -> Clear Invoice
- n=7985 | len=6 | Create Purchase Order Item -> Record Goods Receipt -> Vendor creates invoice -> Record Invoice Receipt -> Remove Payment Block -> Clear Invoice
- n=5298 | len=2 | Create Purchase Order Item -> Delete Purchase Order Item
- n=4244 | len=6 | Create Purchase Order Item -> Receive Order Confirmation -> Vendor creates invoice -> Record Goods Receipt -> Record Invoice Receipt -> Clear Invoice

## 8. Automatic methodological recommendation

- Nu există dovadă suficientă start+complete pentru a interpreta timestampurile drept processing/service time. În Faza 2 vom trata doar cycle time și inter-event delay ca observabile directe.
- Câmpul de resursă este observat cu acoperire 100.00%. Putem analiza workload/resource assignment, dar nu vom presupune capacitate sau ore lucrate fără o reconstrucție explicită.

**Important:** acest raport decide doar ce observabile sunt legitime. Formularea finală a modelului Mathematics/DRO se fixează după revizuirea acestor outputs.
