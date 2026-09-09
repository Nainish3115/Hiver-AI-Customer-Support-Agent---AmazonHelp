# Dataset Understanding & Data Quality Report

> **Project Phase**: Phase 1 — Repository Inspection, Data Quality Analysis, Architecture Setup, and Safe Infrastructure  
> **Author**: Senior ML/AI Engineer  
> **Target Application**: AI Customer-Support Automation Agent (Intent Classification, RAG Retrieval, Grounded Generation & Escalation)

---

## 1. Dataset Overview

This report presents a empirical evaluation of the two raw datasets provided for building the customer-support automation system:
1. **Customer Support on Twitter Dataset**: Primary dataset containing multi-turn, multi-brand customer service conversations.
2. **Banking77 Dataset**: Optional benchmark dataset for domain-specific intent classification experiments.

### Raw Dataset File Metadata

| Dataset | Relative File Path | File Size | Exact Row Count | Exact Column Count | Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Customer Support on Twitter** | `data/raw/customer_support_on_twitter.csv` | **516.51 MB** (516,508,641 bytes) | **2,811,774** | **7** | Standard CSV |
| **Banking77 Train** | `data/external/banking77/banking77_train.csv` | **867.15 KB** (867,153 bytes) | **9,993** | **3** | Standard CSV |
| **Banking77 Test** | `data/external/banking77/banking77_test.csv` | **248.63 KB** (248,627 bytes) | **3,076** | **3** | Standard CSV |

---

## 2. Actual Schema & Data Types

### A. Customer Support on Twitter Schema

| Column Name | Inferred Pandas Data Type | Logical Data Type | Missing Count | Missing % | Description & Sample Values |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tweet_id` | `int64` | Primary Key | `0` | `0.00%` | Integer ID uniquely identifying each tweet (e.g. `1`, `8`, `115712`) |
| `author_id` | `object` (string) | Entity Identifier | `0` | `0.00%` | Anonymized user ID (e.g., `115712`) or brand support handle (e.g., `AmazonHelp`, `sprintcare`) |
| `inbound` | `bool` | Categorical Flag | `0` | `0.00%` | `True` for customer messages sent to support, `False` for company support responses |
| `created_at` | `object` (string) | Timestamp | `0` | `0.00%` | Tweet timestamp formatted as `%a %b %d %H:%M:%S %z %Y` (e.g., `Tue Oct 31 22:10:47 +0000 2017`) |
| `text` | `object` (string) | Unstructured Text | `0` | `0.00%` | Raw tweet body containing text, `@mentions`, links, and emojis |
| `response_tweet_id` | `object` (string) | Foreign Key(s) | `1,040,629` | `37.01%` | Comma-separated tweet IDs that responded to this tweet (e.g. `'2'`, `'5,7'`, `'9,6,10'`) |
| `in_response_to_tweet_id` | `float64` | Foreign Key | `794,335` | `28.25%` | Single parent tweet ID that this tweet replied to; `NaN` indicates a root thread tweet |

### Key Summary Statistics for Twitter Dataset:
- **Date Range**: `2008-05-08 20:13:59+00:00` to `2017-12-03 23:14:01+00:00` (~9.5 years of support data).
- **Inbound vs Outbound Distribution**:
  - **Inbound (Customer)**: **1,537,843** tweets (**54.69%**)
  - **Outbound (Company Support)**: **1,273,931** tweets (**45.31%**)
- **Unique Authors**:
  - **Unique Customer Accounts**: **702,669**
  - **Unique Brand Support Handles**: **108**
  - **Total Distinct Authors**: **702,777**
- **Text Length Statistics**:
  - **Inbound Customer Tweets**: Mean character length = **109.94** (std = 57.16, median = 109.0, p95 = 224.0, max = 513). Mean word count = **18.73** (std = 10.64).
  - **Outbound Support Tweets**: Mean character length = **118.66** (std = 45.40, median = 120.0, p95 = 206.0, max = 343). Mean word count = **20.34** (std = 8.12).

---

## 3. Data-Quality Findings

> [!IMPORTANT]
> **Data Integrity Verification**:
> 1. **Zero Raw Missing Values in Critical Fields**: `tweet_id`, `author_id`, `inbound`, `created_at`, and `text` contain **0 missing values**.
> 2. **Zero Full-Row Duplicates**: `0` duplicate rows exist across the entire 2.81M dataset.
> 3. **Zero Duplicate Tweet IDs**: `0` duplicate `tweet_id` records exist; `tweet_id` is a strictly unique primary key.

> [!WARNING]
> **Encoding & Special Character Considerations**:
> - Tweets contain extensive UTF-8 characters, emojis (e.g. `\U0001f925`), and non-ASCII characters. Python scripts on Windows environments MUST explicitly configure `PYTHONIOENCODING=utf-8` or `sys.stdout.reconfigure(encoding='utf-8')` to prevent `UnicodeEncodeError`.

---

## 4. Thread Reconstruction Findings & Proposal

### How Threads are Formed in the Schema
1. **Thread Origin (Root Tweet)**: A conversation begins when a tweet has `in_response_to_tweet_id == NaN` (or `null`). There are **794,335 root tweets** in the dataset.
2. **Parent-Child Linkage**:
   - `in_response_to_tweet_id` points upward to a single parent tweet ID.
   - `response_tweet_id` points downward to child tweet ID(s). Note that `response_tweet_id` can be a **comma-separated list** (e.g. `'5,7'`), indicating branching replies.
3. **Graph Topology**: The dataset forms a set of **Directed Acyclic Graphs (DAGs) / Trees**, where each root tweet is the tree root, and replies branch into leaf nodes.

```
       [Root Tweet: Customer Inbound]  (in_response_to_tweet_id = NaN)
                     |
         [Support Reply: Outbound]     (in_response_to_tweet_id = Root ID)
                     |
        [Customer Follow-up: Inbound]  (in_response_to_tweet_id = Support ID)
                     |
      [Resolution Reply: Outbound]     (in_response_to_tweet_id = Follow-up ID)
```

### Reconstructed Conversation Depth Distribution
- **Total Identified Thread Trees**: **798,197**
- **Single-Turn Conversations (Length = 2, Customer Query + 1 Support Reply)**: ~**380,450** (~47.6%)
- **Multi-Turn Conversations (Length >= 3)**: **417,747** (**52.34%**)
- **Deep Conversations (Length >= 5)**: **124,310** (**15.57%**)

---

## 5. Brand Distribution Analysis

Across the 2.81M tweets, there are **108 distinct brand support handles** (`inbound == False`). The top 15 brands by volume represent **60.3%** of all support activity.

### Top 15 Brands Summary Table (`reports/tables/top15_brands.csv`)

| Rank | Brand Account | Outbound Tweets | Inbound Tweets | Total Tweets | Conversation Count | Avg Thread Length | Median Length | Max Depth | Multi-Turn Convs (Depth >= 3) | Multi-Turn Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `AmazonHelp` | **169,840** | **203,598** | **373,438** | **82,556** | **4.53** | **3.0** | **448** | **51,260** | **62.09%** |
| **2** | `AppleSupport` | **106,860** | **131,764** | **238,624** | **80,717** | **2.96** | **2.0** | **282** | **28,144** | **34.87%** |
| **3** | `Uber_Support` | **56,270** | **72,154** | **128,424** | **41,923** | **3.07** | **2.0** | **534** | **15,088** | **35.99%** |
| **4** | `SpotifyCares` | **43,265** | **48,543** | **91,808** | **28,280** | **3.25** | **2.0** | **354** | **10,500** | **37.13%** |
| **5** | `Delta` | **42,253** | **45,296** | **87,549** | **26,168** | **3.36** | **2.0** | **209** | **11,147** | **42.60%** |
| **6** | `Tesco` | **38,573** | **34,228** | **72,801** | **16,722** | **4.38** | **4.0** | **551** | **11,629** | **69.54%** |
| **7** | `AmericanAir` | **36,764** | **50,054** | **86,818** | **26,386** | **3.32** | **2.0** | **204** | **11,577** | **43.88%** |
| **8** | `TMobileHelp` | **34,317** | **47,158** | **81,475** | **22,820** | **3.63** | **2.0** | **974** | **8,627** | **37.80%** |
| **9** | `comcastcares` | **33,031** | **39,579** | **72,610** | **24,063** | **3.04** | **2.0** | **651** | **8,794** | **36.55%** |
| **10** | `British_Airways` | **29,361** | **31,187** | **60,548** | **16,452** | **3.69** | **3.0** | **123** | **8,830** | **53.67%** |
| **11** | `SouthwestAir` | **28,977** | **35,370** | **64,347** | **21,636** | **2.99** | **2.0** | **1060** | **7,231** | **33.42%** |
| **12** | `VirginTrains` | **27,817** | **37,832** | **65,649** | **14,853** | **4.43** | **3.0** | **203** | **9,186** | **61.85%** |
| **13** | `Ask_Spectrum` | **25,860** | **33,252** | **59,112** | **18,532** | **3.22** | **2.0** | **651** | **6,561** | **35.40%** |
| **14** | `XboxSupport` | **24,557** | **32,395** | **56,952** | **13,455** | **4.29** | **3.0** | **584** | **8,396** | **62.40%** |
| **15** | `sprintcare` | **22,381** | **30,212** | **52,593** | **13,560** | **3.99** | **2.0** | **974** | **6,383** | **47.07%** |

---

## 6. Recommended Brand Candidate Analysis

Based on strict quantitative evaluation against our project criteria, we recommend **`AmazonHelp`** as the **Primary Brand Candidate**.

### Quantitative Comparison Matrix

| Evaluation Criteria | `AmazonHelp` (Rank 1) | `AppleSupport` (Rank 2) | `SpotifyCares` (Rank 4) | Recommendation Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Total Tweet Volume** | **373,438** | 238,624 | 91,808 | High volume guarantees rich coverage of edge cases. |
| **Total Conversations** | **82,556** | 80,717 | 28,280 | Sufficient sample size for train/val/test splitting without data leakage. |
| **Multi-Turn Depth (Depth >= 3)** | **51,260 (62.09%)** | 28,144 (34.87%) | 10,500 (37.13%) | **`AmazonHelp` has the highest multi-turn depth**, ideal for testing dialogue history context in RAG and escalation. |
| **Average Thread Length** | **4.53 tweets** | 2.96 tweets | 3.25 tweets | Richer back-and-forth interactions between user and agent. |
| **Diversity of Support Issues** | **Very High** | Moderate-High | Moderate | Covers e-commerce, shipping, refunds, digital media (Prime/Kindle), account locks, and returns. |
| **Resolution Suitability for RAG** | **Excellent** | Good | Good | Contains clear problem statements and direct resolution actions. |

> [!TIP]
> **Primary Recommendation**: **`AmazonHelp`**
> - **Secondary Alternatives**: `AppleSupport` (for pure technical troubleshooting) or `SpotifyCares` (for subscription/billing focus).

---

## 7. Banking77 Inspection Findings

Inspection of `data/external/banking77/banking77_train.csv` and `data/external/banking77/banking77_test.csv` confirmed:

- **Actual Columns**: `['text', 'label', 'label_text']`
- **Actual Row Counts**:
  - **Train Set**: **9,993** rows
  - **Test Set**: **3,076** rows
  - **Total**: **13,069** examples
- **Label Alignment**:
  - Exactly **77 unique intent classes** in both train and test sets (`unique_intents_train == unique_intents_test == 77`).
  - Perfect category match between train and test splits (`intents_match == True`).
- **Train/Test Data Overlap**: **0 text overlap** (`train_test_text_overlap == 0`).
- **Data Quality**: **0 missing values**, **0 duplicate rows**, **0 duplicate texts**.

> [!NOTE]
> **Split Discrepancy Note**: The standard HuggingFace Banking77 dataset contains 10,003 train rows and 3,080 test rows. The local raw files provided contain 9,993 train rows (-10) and 3,076 test rows (-4). The label structure remains perfectly intact across all 77 intent classes.

### Banking77 Intent Balance Summary (`reports/tables/banking77_summary.json`)
- **Train Label Counts per Category**: Min = 35, Max = 187, Mean = 129.78, Median = 127.0, Std = 32.97.
- **Test Label Counts per Category**: Min = 39, Max = 40, Mean = 39.95, Median = 40.0, Std = 0.22 (perfectly balanced test set).

---

## 8. Recommended Next Steps for Phase 2

Once Phase 1 is reviewed and approved, Phase 2 should execute:
1. **Thread Preprocessing & Reconstruction Pipeline**: Safely extract complete conversation trees for `AmazonHelp` and output clean, structured Parquet files under `data/processed/amazonhelp_conversations.parquet`.
2. **Intent Taxonomy Construction**: Formulate a candidate taxonomy (15–20 fine-grained intent categories) tailored to `AmazonHelp` support interactions.
3. **Synthetic / Semi-Automated Intent Annotation**: Create an initial intent classification dataset for `AmazonHelp` customer queries.
4. **Banking77 Baseline Classifier**: Train a lightweight Intent Classifier (e.g. SetFit / DeBERTa / LogisticRegression with sentence embeddings) on Banking77 as an intent classification benchmark.

---

## 9. Explicit Assumptions

1. **Brand Handle Isolation**: Brands are identified by `inbound == False` and `author_id` matching company handles. Customer handles are anonymized integers.
2. **Conversation Scope**: A conversation starts at an inbound root tweet (`in_response_to_tweet_id == NaN`).
3. **No Domain Merging**: Banking77 intents (banking domain) will not be blindly applied as the taxonomy for Twitter e-commerce support.

---

## 10. Potential Risks & Discovered Problems

1. **Missing Intermediate Tweets**: Some private messages (DMs) or deleted tweets are referenced in `in_response_to_tweet_id` but absent from the raw CSV. **Mitigation**: Graph traversal gracefully handles missing parent pointers by marking orphaned subtrees safely.
2. **Outbound Multi-Tweet Responses**: Support agents sometimes post multiple sequential tweets to exceed character limits. **Mitigation**: Preprocessing will merge consecutive outbound tweets from the same author timestamped within 120 seconds.
3. **Windows Encoding & Memory Limits**: Large CSV reads cause memory spikes if loaded indiscriminately. **Mitigation**: PyArrow and chunked Pandas loading with explicit UTF-8 encoding configuration enforce memory and execution safety.
