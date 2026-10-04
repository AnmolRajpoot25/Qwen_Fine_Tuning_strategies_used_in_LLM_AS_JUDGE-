# LLM-as-a-Judge — Fine-Tuning Qwen2.5-7B as a Technical Answer Judge

Fine-tune **Qwen2.5-7B-Instruct** into a pairwise **technical answer judge** using **QLoRA** on
consumer GPUs (Colab T4 / Kaggle 2×T4), then measure whether the judge is actually *reliable* —
not just accurate.

The core idea: a judge that always says "A" scores 100% if you never swap the answers. So every
benchmark in this repo evaluates **each pair twice, in both orders**, and reports
**position consistency** and **position bias** alongside accuracy.

- **Weights:** [`Anmol2507/LLM_as_judge_fine_tuned_Qwen_7B`](https://huggingface.co/Anmol2507/LLM_as_judge_fine_tuned_Qwen_7B) (LoRA adapter)
- **Base model:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Method:** 4-bit NF4 QLoRA + LoRA, DDP on 2 GPUs, assistant-only loss, position-swap augmentation

---

## Table of Contents

- [Why this project](#why-this-project)
- [Results](#results)
- [The two judge formats](#the-two-judge-formats)
- [Repository layout](#repository-layout)
- [Pipeline overview](#pipeline-overview)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation harness](#evaluation-harness)
- [Training recipe](#training-recipe)
- [Dataset construction](#dataset-construction)
- [Anti-position-bias design](#anti-position-bias-design)
- [Configuration](#configuration)
- [Hardware notes](#hardware-notes)
- [Project status](#project-status)
- [License](#license)

---

## Why this project

Public LLM-judge benchmarks usually report a single accuracy number on a fixed A/B layout. That
number is trivially gameable by position bias, verbosity bias, or self-preference, and it says
nothing about whether the judge is *stable*.

This repo treats judge quality as three separate properties:

| Property | How it is measured |
|---|---|
| **Accuracy** | Does the judge pick the better answer? |
| **Position invariance** | Does the verdict survive swapping Answer A and Answer B? |
| **Calibrated reliability** | Are retries, latency, and self-reported confidence sane? |

Concretely, that means: swap-augmented training data, split-before-augment (to avoid leakage),
dual-order benchmarking, per-category and per-failure-type breakdowns, and a failure-analysis pass
that prints the raw scores of every inconsistent pair.

---

## Results

Baseline evaluation of **Qwen2.5-7B-Instruct** (no fine-tuning), 4-bit NF4, greedy decoding,
`max_new_tokens=300`, single T4.

### Hard technical benchmark — 100 pairs / 200 evaluations

| Metric | Value |
|---|---|
| Original-order accuracy | **0.99** |
| Reversed-order accuracy | **0.95** |
| **Position consistency** | **0.94** |
| Position bias rate | 0.02 |
| First-position / second-position bias | 1 / 1 |
| Retry rate | 0.00 |
| Avg latency | 15.46 s (min 12.52 s, max 25.40 s) |
| Avg self-reported confidence | 0.871 |

### Position consistency by category

| Category | Pairs | Original | Reversed | **Consistency** |
|---|---:|---:|---:|---:|
| DSA | 20 | 1.000 | 0.900 | **0.900** |
| Competitive Programming | 15 | 0.933 | 0.933 | **0.867** |
| C++ / Python | 10 | 1.000 | 1.000 | **1.000** |
| SQL / DBMS | 10 | 1.000 | 1.000 | **1.000** |
| OS / Systems | 10 | 1.000 | 1.000 | **1.000** |
| Backend / System Design | 10 | 1.000 | 1.000 | **1.000** |
| AI / ML | 8 | 1.000 | 0.875 | **0.875** |
| LLM / RAG / Agents | 8 | 1.000 | 1.000 | **1.000** |
| Frontend | 4 | 1.000 | 0.750 | **0.750** |
| DevOps / Cloud | 5 | 1.000 | 1.000 | **1.000** |

### Failure analysis (6 inconsistent pairs out of 100)

| Failure type | Count |
|---|---:|
| `logic_error` | 2 |
| `off_by_one` | 1 |
| `complexity` | 1 |
| `data_leakage` | 1 |
| `duplicate_request` | 1 |

The dominant failure mode is **not** position bias — it is the judge scoring *both* answers as
perfect and returning `TIE`. Example (`hard_dsa_001`, "two-sum with distinct indices"): in the
reversed order the judge gave **10/10 on both candidates**, versus 10.0 vs 8.4 in the original
order. This is a **discrimination failure**, not a positional one, and it is the thing to fix
next.

An earlier 50-pair technical benchmark and a 10-pair position-bias suite are also included in the
notebooks (consistency 0.98 and 0.90 respectively).

> These numbers are the **base-model baseline**. Post-fine-tuning benchmark numbers are not
> published in this repo — see [Project status](#project-status).

---

## The two judge formats

The repo contains two different judge contracts. They are **not** interchangeable.

### 1. Rubric judge — `src/judge/`

Model scores each answer on five criteria, `0–10`, and emits strict JSON. Deterministic
post-processing derives the winner, weighted score, and confidence.

```json
{
  "A": {"correctness": 0, "relevance": 0, "completeness": 0, "reasoning": 0, "clarity": 0, "feedback": "..."},
  "B": {"correctness": 0, "relevance": 0, "completeness": 0, "reasoning": 0, "clarity": 0, "feedback": "..."}
}
```

Weights (`JudgeEvaluator.CRITERIA`):

| Criterion | Weight |
|---|---:|
| correctness | **0.40** |
| relevance | 0.20 |
| completeness | 0.15 |
| reasoning | 0.15 |
| clarity | 0.10 |

The model is explicitly told **not** to output `final_score`, `winner`, or `confidence` —
`src/judge/evaluator.py:213` computes them. Confidence is derived from the score gap
(`|Δ| ≥ 2.0 → 0.90`, `≥ 1.0 → 0.75`, `≥ 0.5 → 0.60`, else `0.50`).

### 2. Arena judge — training target

The published adapter was trained on the compact pairwise format used by Chatbot Arena:

```json
{"winner": "A"}
```

with `winner ∈ {A, B, tie}` and a system prompt that instructs the model to ignore the identity of
the models that produced the answers.

> **Integration note:** the LoRA adapter in the linked HF repo emits `{"winner": ...}`, while
> `JudgeEvaluator` expects the five-criteria rubric JSON. To use the adapter with `src/judge/`,
> retrain on the rubric format (`src/training/formatter.py` already emits it) or add a parser for
> the arena format.

---

## Repository layout

```
.
├── configs/
│   ├── dataset_config.py            # paths, val ratio, seed, supported categories
│   ├── dataset_distribution.json    # per-domain example budget + difficulty/relation mix
│   ├── failure_distribution.json    # target mix of injected failure types
│   ├── generation_config.json       # providers, retries, cooldown, checkpoint interval
│   └── generation_tasks.json        # 1,500 planned generation tasks (id/category/difficulty/…)
├── src/
│   ├── judge/
│   │   ├── prompts.py               # JUDGE_SYSTEM_PROMPT (rubric, bias-resistant)
│   │   └── evaluator.py             # JudgeEvaluator: generation, JSON repair, retry, scoring
│   ├── model/
│   │   └── loader.py                # ModelLoader: 4-bit NF4 causal-LM loader
│   ├── evaluation/
│   │   ├── benchmark.py             # JudgeBenchmark: flat expected_winner accuracy
│   │   ├── technical_benchmark.py   # 50-pair dual-order benchmark
│   │   ├── hard_benchmark.py        # 100-pair dual-order benchmark + category/failure metrics
│   │   ├── position_bias.py         # PositionBiasTester: bias-type classification
│   │   ├── reliability.py           # ReliabilityAnalyzer: score/winner consistency, retries
│   │   └── metrics.py               # BenchmarkMetrics: accuracy rollups
│   ├── training/
│   │   ├── schema.py                # validate_training_example / candidate score bounds
│   │   ├── augment.py               # position-swap augmentation
│   │   ├── formatter.py             # rubric prompt → chat messages SFT records
│   │   ├── dataset_processor.py     # load → validate → dedupe → split → augment → format
│   │   ├── prepare_dataset.py       # CLI entry point for the above
│   │   ├── generation_prompts.py    # synthetic-example generation prompt
│   │   ├── data_generation_schema.py# per-domain topics + failure types
│   │   └── answer_variants.py       # excellent → hallucinated quality ladder
│   └── utils/
├── tests/                           # reserved for judge + inference tests
├── checkpoints/                     # generation checkpoint + provider health state
├── Judging_LLMs_.ipynb              # Colab: eval harness, data generation, Drive sync, HF push
├── Judging_LLMs_Colab.ipynb         # Same pipeline, Colab-tuned copy
└── llmas-judge.ipynb                # Kaggle: 2×T4 DDP QLoRA training + benchmark + push
```

---

## Pipeline overview

```
                  ┌──────────────────────────────────────────┐
  Stage 1         │ Synthetic data generation               │
  DATA            │ Gemini / Groq / OpenRouter router        │
                  │ → retries, cooldowns, disk checkpoints   │
                  │ → validated against schema.py            │
                  └───────────────────┬──────────────────────┘
                                      │
                  ┌───────────────────▼──────────────────────┐
                  │ Public data                              │
                  │ potsawee/chatbot-arena-llm-judges        │
                  │ + 10k sampled code-edit preference pairs │
                  └───────────────────┬──────────────────────┘
                                      │
                  ┌───────────────────▼──────────────────────┐
                  │ DatasetProcessor                        │
                  │ validate → dedupe → split 90/10          │
                  │ → position-swap augment → chat format    │
                  └───────────────────┬──────────────────────┘
                                      │
                  ┌───────────────────▼──────────────────────┐
  Stage 2         │ QLoRA fine-tune (2×T4, DDP)              │
  TRAIN           │ assistant-only loss, max_len 768         │
                  │ → Drive-synced checkpoints every 250 steps│
                  └───────────────────┬──────────────────────┘
                                      │
                  ┌───────────────────▼──────────────────────┐
  Stage 3         │ Dual-order benchmark + reliability       │
  EVAL            │ accuracy / consistency / bias / latency  │
                  │ → failure analysis on inconsistent pairs│
                  └───────────────────┬──────────────────────┘
                                      │
                  ┌───────────────────▼──────────────────────┐
  Stage 4         │ Adapter → Hugging Face Hub               │
  PUBLISH         │ base_model_name_or_path rewritten to HF ID│
                  └──────────────────────────────────────────┘
```

---

## Setup

### Requirements

```bash
pip install -q transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 \
                bitsandbytes datasets huggingface_hub \
                google-genai groq openai python-dotenv tenacity tqdm
```

Optional extras, depending on the stage you run:

| Need | Package |
|---|---|
| Synthetic data generation | `google-genai`, `groq`, `openai`, `tenacity`, `tqdm` |
| Training / evaluation | `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets` |
| Google Drive checkpoint sync | `google-api-python-client`, `google-auth`, `google-auth-oauthlib` |
| Notebook tables | `pandas` |

### API keys

Never commit keys. In the notebooks they are read interactively via `getpass`, so they stay in the
session only:

```python
import os, getpass
os.environ["GEMINI_API_KEY"]     = getpass.getpass("Gemini: ")
os.environ["GROQ_API_KEY"]       = getpass.getpass("Groq: ")
os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OpenRouter: ")
os.environ["OPENROUTER_MODEL"]   = "openrouter/free"
```

`credentials.json` and `env.example` are git-ignored placeholders — fill them locally, never commit.

### Base model

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="models/Qwen2.5-7B-Instruct",
)
```

---

## Usage

### 1. Load the model and judge

```python
import sys
sys.path.insert(0, "/content/drive/MyDrive/LLM-Judge")

from src.model.loader import ModelLoader
from src.judge import JudgeEvaluator

tokenizer, model = ModelLoader("Qwen/Qwen2.5-7B-Instruct").load()

judge = JudgeEvaluator(model=model, tokenizer=tokenizer)

outcome = judge.evaluate(
    problem="Explain why a hash set gives O(n) two-sum and where the check must happen.",
    answer_a="Insert and check in one pass over the array, testing membership before insertion.",
    answer_b="Sort the array, then use two pointers scanning from both ends.",
)

print(outcome["result"]["winner"])      # 'A' | 'B' | 'TIE'
print(outcome["result"]["confidence"])  # 0.50 – 0.90
print(outcome["attempts"])              # JSON-repair retries consumed
```

### 2. Load the fine-tuned adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "Anmol2507/LLM_as_judge_fine_tuned_Qwen_7B"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()
```

### 3. Prepare a training dataset

```bash
python -m src.training.prepare_dataset
```

Writes `train.jsonl`, `validation.jsonl`, and `statistics.json` into `PROCESSED_DATA_DIR`, and
prints a load → validate → dedupe → split → augment → format report.

### 4. Fine-tune (Kaggle, 2×T4)

`llmas-judge.ipynb` writes `train_qwen_ddp.py` and launches it:

```bash
torchrun --nproc_per_node=2 /kaggle/working/train_qwen_ddp.py
```

### 5. Benchmark

```python
from src.evaluation.hard_benchmark import HardTechnicalBenchmark

bench = HardTechnicalBenchmark(
    judge=judge,
    dataset_path="datasets/benchmark/hard_technical_100.json",
)

results = bench.run()                                    # 200 evaluations
print(bench.calculate_metrics(results))                 # accuracy / consistency / bias / latency
print(bench.category_metrics(results))                   # per-domain consistency
print(bench.failure_type_metrics(results))               # per-failure-type consistency
```

---

## Evaluation harness

All benchmark classes are judge-agnostic — pass any object exposing
`.evaluate(problem, answer_a, answer_b)`.

| Module | Purpose |
|---|---|
| `JudgeBenchmark` | Single-order accuracy against `expected_winner`; smoke test |
| `TechnicalBenchmark` | 50 dual-order pairs; accuracy, consistency, bias, latency |
| `HardTechnicalBenchmark` | 100 dual-order pairs + category and failure-type breakdowns |
| `PositionBiasTester` | Classifies each pair as `none` / `first_position` / `second_position` / `inconsistent` / `tie_or_other` |
| `ReliabilityAnalyzer` | Cross-order score drift → `score_consistency`; retry rate, latency percentiles, mean confidence |
| `BenchmarkMetrics` | Accuracy rollups by category |

Metric definitions:

- **original_accuracy** — fraction of pairs where the good answer won in slot A.
- **reversed_accuracy** — fraction of pairs where the good answer won in slot B.
- **position_consistency** — fraction of pairs where **both** orders were correct. This is the
  number to optimise: it cannot be inflated by position bias.
- **position_bias_rate** — fraction of pairs where the judge chose the *same slot* in both orders.
- **score_consistency** — `1 − mean|Δ final score| / 10`, where Δ is measured per *answer* across
  the two orderings (original A is compared against reversed B).

### Benchmark dataset schema

```json
{
  "id": "hard_dsa_001",
  "category": "DSA",
  "difficulty": "hard",
  "failure_type": "logic_error",
  "problem": "...",
  "good_answer": "...",
  "bad_answer": "..."
}
```

`good_answer` is always expected in slot A, so the reversed pass flips the expectation to `B`.

### Output robustness

Small instruct models do not reliably emit clean JSON, so `JudgeEvaluator` layers four defences:

1. Direct `json.loads`.
2. Strip markdown fences (```` ```json ```` … ```` ``` ````).
3. Regex-extract the outermost `{...}` block.
4. **Retry** with an appended corrective user message restating the exact schema.

Every attempt is counted and surfaced as `attempts`; `raw_response` is returned for offline
debugging. Scores are then schema-validated (integer, `0–10`, all five criteria present,
`feedback` is a string) before scoring.

---

## Training recipe

QLoRA on `Qwen2.5-7B-Instruct`, as executed by `llmas-judge.ipynb`:

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quant, fp16 compute |
| LoRA `r` / `alpha` / `dropout` | 8 / 16 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optimizer | `paged_adamw_8bit` |
| Learning rate | `1e-4`, 300 warmup steps |
| Batch size | 1 per device × 8 grad accumulation × 2 GPUs |
| Epochs | 1 (≈1,563 optimizer steps) |
| Max sequence length | 768 |
| Gradient checkpointing | enabled |
| Precision | fp16 |
| `use_cache` | disabled during training |
| Parallelism | `torchrun --nproc_per_node=2` (DDP) |
| Checkpointing | every 250 steps, mirrored to Google Drive |

### Assistant-only loss

This is the detail that most often breaks SFT-for-JSON work, so it is worth stating explicitly.
`tokenize_example` tokenizes the prompt with `add_generation_prompt=True` and the full conversation
with `add_generation_prompt=False`, then masks everything before the assistant turn with `-100`:

```python
labels = [-100] * len(prompt_part) + response_ids
```

Two safety checks reject degenerate examples rather than silently training on nothing:

- `len(response_ids) == 0` → `ValueError` (no assistant turn to supervise).
- `all(label == -100)` → `ValueError` (no trainable tokens).

Truncation is **response-preserving**: if the response fits, the prompt is trimmed from the *left*
(keeping the most recent context); if the response alone exceeds the budget, the response is
truncated instead. Padding is dynamic, `pad_to_multiple_of=8`, with labels padded using `-100`.

### Crash-resilient checkpointing

Kaggle sessions are ephemeral, so a `TrainerCallback` mirrors every checkpoint to Google Drive with
a resumable chunked upload, flattens paths (`a/b` → `a__b`), and writes a `manifest.json`. A
`_COMPLETE` marker is uploaded **last** — on restart the script scans Drive, ignores any
checkpoint without that marker, and resumes from the newest complete one. Only rank 0 uploads, with
a `torch.distributed.barrier()` afterwards so all ranks continue together.

---

## Dataset construction

### Synthetic technical pairs

`configs/generation_tasks.json` holds **1,500 planned tasks**, each a
`(id, category, subcategory, difficulty, answer_relation, failure_type)` tuple. Budget:

| Domain | Count | | Domain | Count |
|---|---:|---|---|---:|
| DSA | 250 | | Backend / System Design | 180 |
| Competitive Programming | 200 | | LLM / RAG / Agents | 180 |
| AI / ML | 150 | | C++ / Python | 120 |
| SQL / DBMS | 120 | | OS / Systems | 100 |
| Frontend | 100 | | DevOps / Cloud | 100 |
| | | | **Total** | **1,500** |

Difficulty mix: 15% easy / 30% medium / 35% hard / 20% expert.

Answer pairs are drawn from a quality ladder (`src/training/answer_variants.py`): `excellent`,
`correct_but_incomplete`, `partially_correct`, `subtle_error`, `incorrect`, `hallucinated` — so a
quarter of pairs are genuinely close calls rather than obvious wins.

Failure types are injected deliberately (`configs/failure_distribution.json`): `logic_error` 15%,
`edge_case` 12%, `complexity` 12%, `constraint_reasoning` 10%, `off_by_one` 8%, and so on. The
prompt insists that wrong answers stay *plausible* and that good answers do not all get 10/10.

A provider router dispatches across Gemini, Groq, and OpenRouter with per-provider success/failure
tracking, latency accounting, exponential cooldowns on 429/503, disk checkpoints, and resume from
the last completed task ID. Last recorded run: **605 generated, 198 failed** (rate limits dominated).

### Public data

- `potsawee/chatbot-arena-llm-judges` — real human pairwise preferences, converted to the
  `question / answer_a / answer_b / winner` schema.
- A code-edit preference dataset — 10,000 examples sampled, with **A/B position randomized** so the
  chosen response is not always in slot A.

The two are concatenated, shuffled with seed 42, and split 90/10.

### Cleaning pipeline

`DatasetProcessor.process()` runs: load all `*.json` → schema-validate → dedupe on
`(problem, answer_a, answer_b)` → split → augment → format. Statistics are written to
`statistics.json` for every stage, and invalid examples are reported with their IDs and reason.

---

## Anti-position-bias design

Position bias is attacked from three directions.

**1. Prompt-level (`src/judge/prompts.py`).** The system prompt states that A and B carry no
meaning, explicitly forbids preferring the first, second, longer, or shorter answer, requires
*independent* per-answer scoring before any comparison, and adds a pre-submission checklist
("would I give this exact answer the same scores in the other position?"). It also separates
*completeness* from *verbosity* — brevity alone must not lose points.

**2. Data-level (`src/training/augment.py`).** Every training example is duplicated with A and B
swapped and the evaluation block swapped to match, so the model sees each pair in both orders
during training. Critically, `DatasetProcessor` **splits before augmenting** — augmenting first
would leak a pair's swapped twin across the train/validation boundary and inflate every metric.

**3. Evaluation-level.** Every benchmark runs both orders and reports consistency and bias
separately from accuracy.

---

## Configuration

| File | Controls |
|---|---|
| `configs/dataset_config.py` | `PROJECT_ROOT`, raw/processed dirs, `VALIDATION_RATIO` (0.10), `RANDOM_SEED` (42), supported categories |
| `configs/dataset_distribution.json` | Per-domain example budget, difficulty mix, answer-relation mix |
| `configs/failure_distribution.json` | Target distribution of injected failure types |
| `configs/generation_config.json` | Providers, `max_retries`, retry delay, examples per domain, checkpoint interval |
| `configs/generation_tasks.json` | The 1,500 concrete generation tasks |
| `configs/judge.yaml`, `configs/model.yaml` | Reserved — currently empty |

`configs/dataset_config.py` ships with absolute Colab paths (`/content/drive/MyDrive/LLM-Judge`).
**Change `PROJECT_ROOT` to match your environment** before running the dataset pipeline.

---

## Hardware notes

| Stage | Hardware | Notes |
|---|---|---|
| Evaluation | 1× T4 (16 GB) | 4-bit NF4 fits; ~15 s per judgment |
| QLoRA training (Colab) | 1× T4 | `r=16` on all attention + MLP projections, `max_len 1536` |
| QLoRA training (Kaggle) | 2× T4, DDP | `r=8` on attention projections only, `max_len 768` |

Two hard-won details from the Colab run:

- **Do not call `prepare_model_for_kbit_training()`** on the T4 — it triggered a large FP32 memory
  spike. `model.enable_input_require_grads()` alone is enough for gradient checkpointing with
  frozen embeddings.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be set **before** importing `torch`.

---

## Project status

Honest inventory of what is and isn't here yet.

**Implemented and exercised**

- `src/judge/` — rubric judge with JSON repair, schema validation, retry, weighted scoring
- `src/model/loader.py` — 4-bit NF4 loader
- `src/evaluation/` — all six benchmark/analysis modules
- `src/training/` — schema validation, position-swap augmentation, rubric SFT formatter,
  `DatasetProcessor` pipeline, synthetic-generation prompt scaffolding
- `configs/` — distribution, failure-mix, and 1,500-task generation plan
- Kaggle 2×T4 QLoRA training → Drive-synced checkpoints → HF Hub publish
- Baseline benchmark results and failure analysis (see [Results](#results))

**Not yet done**

- **Post-fine-tuning benchmark numbers.** The linked adapter was published, but the dual-order
  benchmark has only been run against the base model. Publishing before/after numbers is the single
  highest-value next step.
- **Placeholder files.** `src/training/train_qwen_qlora.py`, `create_arena_sft.py`,
  `synthetic_generator.py`, `generator_clients.py`, `tests/test_inference.py`,
  `tests/test_judge.py`, `configs/judge.yaml`, `configs/model.yaml`, and `env.example` are all
  **empty**. The logic they were meant to hold currently lives inline in the notebooks.
- **Prompt contract mismatch.** The adapter is trained on the arena `{"winner": ...}` format while
  `JudgeEvaluator` expects the five-criteria rubric JSON (see
  [The two judge formats](#the-two-judge-formats)).
- **Notebook-centric.** `Judging_LLMs_.ipynb` and `Judging_LLMs_Colab.ipynb` are near-duplicates
  with hardcoded Drive folder IDs; they should be consolidated into scripts plus one thin notebook.
- **`gitignore` (no dot) is an empty stray file** alongside the real `.gitignore`; safe to delete.
- Synthetic generation completed 605 / 1,500 tasks before rate limits stopped it — resume is
  supported from `checkpoints/generation_checkpoint.json`.

---

## License

Released under the MIT License. A `LICENSE` file still needs to be added to the repository root.

The fine-tuned weights are derived from
[`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), which is licensed
under the Apache-2.0 license — follow its terms when redistributing the adapter.