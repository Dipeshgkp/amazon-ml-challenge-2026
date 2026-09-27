# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Cloud Ninjas  
**Members:** Dipesh Yadav, Aryan Garad, Suraj Saw, Sujal Muluk (MIT ADT University, Pune)  

---

## Overview

This repository contains the solution and submission package for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

The objective is to resolve noisy business records across three heterogeneous data sources (`source1`, `source2`, `source3`) with significant typographical, transliteration (9 Indic scripts), address formatting, and cross-country noise (including unseen test data from France), evaluated using macro-averaged $F_{0.5}$.

Detailed methodology, error analysis, stress testing, and experimental results are documented in [Documentation_template.md](Documentation_template.md).

---

## Submission Package Layout

```text
Cloud_Ninjas_submission.zip
├── output/
│   ├── matching_results.tsv       # final predicted matches
│   └── candidate_pairs.tsv        # candidate pairs from blocking
├── code/
│   └── business_entity_resolution/
│       ├── src/                   # complete source code (base, v2, v4)
│       ├── README.md              # step-by-step reproduction instructions
│       └── requirements.txt       # pinned environment dependencies
└── Documentation_template.md      # comprehensive methodology write-up
```


---

## Quickstart & Pipeline Reproduction

Detailed reproduction steps and environment requirements are located in [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
