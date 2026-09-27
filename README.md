# High-Scale Multilingual Business Entity Resolution
### Amazon ML Challenge 2026 — Business Entity Resolution Challenge

[![Amazon ML Challenge 2026](https://img.shields.io/badge/Amazon_ML_Challenge-2026-FF9900?logo=amazon&logoColor=white)](https://www.amazon.science/)
[![Track](https://img.shields.io/badge/Track-Business_Entity_Resolution-blue)]()
[![Metric](https://img.shields.io/badge/Evaluation-Macro_F0.5_%3D_0.9832-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Amazon ML Challenge 2026 Solution** — A high-performance, precision-optimized Machine Learning pipeline designed for the **Amazon ML Challenge 2026** to resolve real-world business identities across multi-million row heterogeneous data sources with extreme typographical noise, 9 Indic scripts transliteration, and unseen country domain shifts.

---

## 📌 Problem Statement (PS) — Amazon ML Challenge 2026

In large-scale commercial platforms (such as Amazon Marketplace and supply chain systems), business entity information arrives from independent, unstandardized sources (`Source 1`, `Source 2`, and `Source 3`). These sources share **no common identifiers**, and each record provides partial, noisy, or conflicting fragments of information about real-world businesses.

* **Core Task:** Given deduplicated reference entities from **Source 1**, identify and resolve all corresponding records from **Source 2** and **Source 3**. A Source 1 entity may match zero (singletons), one, or many records from Source 2/3.
* **Key Difficulties & Noise Patterns:**
  * **Multilingual & Transliteration Noise:** ~7% of records are written in **9 different Indic scripts** (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Gurmukhi, Odia), requiring cross-script phoneme matching.
  * **Extreme Name Variations:** Abbreviations (`Corp` vs. `Corporation`, `Pvt` vs. `Private`), legal suffixes, DBA (doing business as) trade names, punctuation shifts, transposed words, and OCR/keyboard digit-for-letter typos (`Preparat0ry`).
  * **Address Inconsistencies:** Missing postal codes or states (~3–4% empty), non-standard abbreviations (`Rd`, `St`, `Marg`), landmark-based descriptions (`Near SBI ATM`), municipal renumbering, and missing unit numbers.
  * **Unseen Country Domain Shift:** The training dataset contains records from the **US** and **India**, while the test set introduces a third country, **France**, completely absent from training.
* **Evaluation Metric:** The task evaluates on **Macro-averaged $F_{0.5}$ score**:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  $F_{0.5}$ places **$2\times$ more weight on Precision than Recall**. In commercial entity resolution, false merges (linking two distinct businesses) are far more damaging than missed links. Correctly identifying singletons (predicting an empty list) earns full credit.

---

## ⚡ Handling the Multi-Million Row Dataset ($12.5\text{M}+$ Records)

### The Scale Bottleneck
* The corpus contains **2.21 million Source 1 entities** and **10.3 million Source 2 & 3 records** (over **12.5 million records** in total, with 7.64M true links).
* A naive pairwise cross-comparison would demand:
  $$\approx 2.21 \times 10^6 \times 10.3 \times 10^6 \approx 2.27 \times 10^{13}\text{ pair comparisons}$$
  Evaluating over **22 trillion pairs** with complex similarity models is computationally impossible.

### Scalable Architecture & System Design
To make multi-million record processing fast, memory-safe, and runnable on commodity hardware (even laptops with 16GB RAM and mid-tier GPUs like RTX 4050):

1. **Zero-Copy Streaming I/O with Polars & Parquet:**
   * Raw TSVs are converted to partitioned Parquet files with fixed string schemas and zero quoting overhead.
   * Data is streamed in chunked batches, avoiding large memory spikes and keeping RAM utilization strictly under control.
2. **GPU-Accelerated Sparse TF-IDF & Dimensionality Compression:**
   * **Country Partitioning:** Blocks queries strictly within countries (no valid cross-country links exist).
   * **Feature Representation:** Builds character 3-grams of space-free core business names and tokenized address components.
   * **CountSketch Compression:** Compresses sparse high-dimensional n-gram vectors down to **1024 dense dimensions**.
   * **GPU FP16 Candidate Search:** Executes batch matrix multiplications on GPU in FP16 to extract an initial top-40 candidate pool per record in milliseconds, followed by exact cosine re-scoring to yield the **top-10 candidates** per record.
   * **Recall Upper Bound:** Retrieves **98.7% of all true links** (Candidate-Restricted Oracle Macro $F_{0.5} = \mathbf{0.99609}$), collapsing the search space from $2.27 \times 10^{13}$ to $\sim 103\text{M}$ pairs with sub-linear complexity.

---

## 🧠 Modeling & Solution Strategy

### 1. Robust Multilingual Normalization
* **Unified Indic Normalization:** A unified Unicode-offset algorithm maps all 9 Indic scripts into a shared phonological phonetic space, preserving consonants, native numerals, and vowel diacritics.
* **Learned Phonetic Dictionary:** A 526-entry translation dictionary (`praivet` $\rightarrow$ `private`, `injiniyaring` $\rightarrow$ `engineering`) extracted strictly from training pairs to bridge transliteration gaps.
* **Core View Extraction:** Cleans names by stripping legal entities (`LLC`, `Pvt Ltd`, `GmbH`), junk prefixes (`M/s`, `Dr`, `Shri`), and noise characters.

### 2. Direct-Evidence Principle (Duplication Invariance)
* Many classical ER pipelines engineer features based on cluster density or sibling agreement (e.g., how many records agree on candidate S1).
* **The Trap:** An earlier sibling-agreement model achieved $0.986$ on validation but collapsed to $0.933$ on test data due to coherent distractor clusters (e.g., business branch networks sharing names but differing in addresses).
* **The Solution:** Our pipeline enforces **direct evidence only** — every feature depends strictly on the relationship between record $i$ and candidate $j$, plus invariant global catalog statistics. It is mathematically invariant to distractor duplication.

### 3. High-Discriminative Feature Engineering (80+ Features)
* **IDF-Weighted Token Agreement:** Rarity weighting calculated from the reference catalog. Matching a rare token (e.g., `"Zalando"`) provides immense confidence, whereas matching common terms (`"Store"`, `"Enterprises"`) is downweighted.
* **Premise vs. Unit Address Logic:** Separates premise/building numbers from unit/suite numbers; detects numeric contradictions while handling missing numbers gracefully.
* **String Metrics:** RapidFuzz token sort, token set, partial ratio, and Levenshtein distances computed across raw, normalized, and core string representations.
* **Candidate Competition & Margin:** Cosine gaps, rank position, and score margins relative to runner-up candidates.

### 4. Two-Stage Stacking Architecture
* **Stage 1 (Pairwise Classifier):** LightGBM model (255 leaves, learning rate 0.05, 1317 trees) trained on 15.5M candidate pairs to score match probability.
* **Stage 2 (Record-Level Stacking):** Evaluates the top candidate for each record using out-of-sample Stage 1 probabilities, runner-up margin, and competition entropy.
* **Assignment:** Assigns record to candidate if $p \ge 0.70$, automatically mapping unlinked records to singletons.

### 5. Domain Adaptation for Unseen Country (France — v4)
* **French Locale Normalization:** Mines unlabelled France test records to build a dedicated normalizer for administrative divisions (department numbers $\leftrightarrow$ region names), street prefixes (`bis`, `ter`, `cours`, `rond-point`), and legal suffixes (`cie`, `ei`).
* **Semi-Supervised Self-Training:** Uses high-confidence pseudo-labels ($p \ge 0.98$ for positive, $p \le 0.02$ for negative) with a conservative threshold ($p \ge 0.90$) to safeguard precision against unseen domain variance.

---

## 📊 Experimental Results

Evaluated on an entity-isolated split (60% Train, 20% Dev, 20% Sealed Confirmation):

| Pipeline Stage | Dev Macro $F_{0.5}$ | Sealed Confirmation $F_{0.5}$ | Notes |
| :--- | :---: | :---: | :--- |
| **Baseline Recipe (B0)** | `0.97785` | `0.97787` | Reference baseline |
| **Duplication-Invariant (C1)** | `0.97791` | — | Removed sibling features |
| **+ Direct Evidence Features (C2)** | `0.98056` | — | IDF tokens & premise numbers |
| **+ 2.5× Scaled Training (C3)** | `0.98201` | — | Increased pair coverage |
| **+ Two-Stage Stacking (C4 Final)** | **`0.98322`** | **`0.98321`** | **Final production model** |

* **Synthetic Stress Testing:** Evaluated against 31,050 simulated branch distractor clusters; achieved **zero decision flips** when distractors were multiplied up to $8\times$.

---

## 🛠️ Repository Structure

```text
├── output/
│   └── matching_results.tsv            # Final test predictions (1.73M entities)
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── prepare_data.py         # TSV to columnar Parquet parser
│       │   ├── normalization.py        # Text & 9 Indic-script transliteration
│       │   ├── retrieval.py            # GPU TF-IDF candidate generation
│       │   ├── candidates.py           # Top-K candidate extraction
│       │   ├── features.py             # 64 base similarity & numeric features
│       │   ├── v2/                     # Stacking, direct-evidence features & stress suite
│       │   │   ├── run_v2.py           # Pipeline runner
│       │   │   ├── featx.py            # 16 direct-evidence features
│       │   │   ├── train_v2.py         # Stage 1 LightGBM classifier
│       │   │   └── stack_v2.py         # Stage 2 record stacking model
│       │   └── v4/                     # France domain adaptation
│       │       ├── fr_locale.py        # French locale normalizer
│       │       ├── fr_pseudo.py        # Semi-supervised pseudo-labeling
│       │       └── assemble.py         # Final hybrid predictions assembler
│       ├── utils/
│       │   └── validate_submission.py  # Integrity and format validator
│       ├── README.md                   # Step-by-step reproduction instructions
│       └── requirements.txt            # Pinned dependencies
├── Documentation_template.md           # In-depth technical methodology document
└── .gitignore                          # Excludes large binaries & raw TSVs
```

---

## 🚀 Reproduction & Setup

### Environment Requirements
* Python 3.10+
* NVIDIA GPU (recommended for candidate retrieval)
* 16 GB RAM

```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r code/business_entity_resolution/requirements.txt
```

For detailed step-by-step execution instructions from raw data to final predictions, refer to [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
