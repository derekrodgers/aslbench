# PLAN.md: aslbench

## 0. Context for the implementing agent

You are implementing a take-home assessment deliverable: a novel benchmark that
measures a meaningful frontier AI model capability. The design work is complete;
this document is the full specification. Do not redesign; where this plan says
"verify X", verify X and adapt within the stated intent.

**The benchmark: aslbench.** A vision language model (VLM) is shown a single
photograph of a hand (or two hands) forming one American Sign Language (ASL)
fingerspelling sign. It must name the character the sign represents: a digit 0-9
or an uppercase letter A-Z (36 classes). The answer is scored by exact match
against the ground-truth class, which is the folder the image came from. This
tests fine-grained visual perception of fingers and gestures, an area where
general-purpose VLMs tend to lag the specialised CNNs usually trained on this
kind of data.

**Why this qualifies:**

- Objectively scored. Ground truth is a fixed 36-way categorical label. A
  prediction is correct if and only if it equals that label. No human judge, no
  LLM judge, anywhere.
- Public authoritative data. The ASL-HG dataset (Pranto et al., *Data in Brief*,
  January 2026, DOI 10.1016/j.dib.2026.112492) is free and credential-free on
  Mendeley Data (https://data.mendeley.com/datasets/j4y5w2c8w9/1).
- Novel. General-purpose frontier VLMs have not, to our knowledge, been
  benchmarked on this exact fingerspelling dataset; prior work on ASL
  recognition uses specialised CNNs, not general VLMs.
- Non-saturated and reproducible. Fingerspelling is genuinely hard for VLMs, so
  scores land well below perfect while staying above zero. The processed dataset
  is built by a seeded script and runs are archived on disk.

**Hard requirements inherited from the project owner (do not violate):**

1. Environment management uses conda. Not venv, not virtualenv, not uv.
2. All benchmark logic (dataset loading, model invocation, scoring) lives in
   plain importable Python modules that the Dash app calls directly. No logic
   trapped inside CLI scripts that the app shells out to.
3. The app's subset selection defaults to empty; a run only starts after a
   deliberate selection.
4. No deployment steps in this plan or in the implementation. Do not add
   Dockerfiles, fly.toml, or similar.
5. The git history is a graded deliverable, but the human manages all commits.
   Do not use git.
6. The class-encoding image filename must never be sent to the model.

---

## 1. Repository layout

```
novel-benchmark/
├── PLAN.md                      # this file
├── BRIEF.md                     # original assessment brief (already present)
├── environment.yml              # conda environment spec
├── providers.yaml               # provider declarations (see Section 5)
├── scripts/
│   └── subset_dataset.py        # thin CLI: raw ASL-HG -> data/processed/
├── src/
│   └── aslbench/
│       ├── __init__.py
│       ├── config.py            # paths, the 36 classes, providers.yaml loading
│       ├── dataset.py           # processed-dataset loading + subset sampling
│       ├── providers/
│       │   ├── __init__.py      # Provider protocol, registry, factory
│       │   ├── openai_compat.py
│       │   ├── anthropic_provider.py
│       │   └── copilot_provider.py
│       ├── prompts/
│       │   ├── __init__.py      # template loading/rendering
│       │   ├── v1_zeroshot.md
│       │   ├── v2_class_list.md
│       │   └── v3_reasoning.md
│       ├── runner.py            # run orchestration, background threads, state files
│       ├── scoring.py           # parsing, categorical metrics, bootstrap CIs
│       ├── figures.py           # shared Plotly builders (app + report)
│       ├── export.py            # Quarto render invocation + file naming
│       └── app/
│           ├── __init__.py
│           ├── __main__.py      # python -m aslbench.app
│           ├── app.py           # Dash app factory + image route
│           ├── layout.py
│           └── callbacks.py
├── report/
│   └── report.qmd               # parameterized Quarto report template
├── data/
│   ├── raw/                     # raw ASL-HG dataset (gitignored; too large)
│   │   └── asl_hg_dataset/<class>/P<p>_<class>_<n>.jpg
│   └── processed/               # seeded subset, one folder per class
│       └── <class>/P<p>_<class>_<n>.jpg
├── runs/                        # one folder per benchmark run
├── exports/                     # rendered PDF/HTML reports
└── tests/
    ├── test_scoring.py
    ├── test_parsing.py
    ├── test_dataset.py
    ├── test_prompts.py
    ├── test_runner.py
    └── test_comparison.py
```

Install the package in development mode (`pip install -e .` via a minimal
`pyproject.toml`) so `aslbench` imports cleanly from scripts, tests, the app, and
the Quarto report.

---

## 2. Conda environment

`environment.yml`:

```yaml
name: aslbench
channels:
  - conda-forge
dependencies:
  - python=3.11
  - dash
  - dash-bootstrap-components
  - plotly
  - pandas
  - numpy
  - pyarrow
  - pyyaml
  - pytest
  - jupyter
  - nbformat
  - nbclient
  - ipykernel
  - papermill
  - pip
  - pip:
      - openai
      - anthropic
      - github-copilot-sdk
```

Setup commands (document these in the README):

```bash
conda env create -f environment.yml
conda activate aslbench
pip install -e .
copilot --version   # verify the Copilot CLI is available
quarto --version    # verify >= 1.4 (PDF export uses its bundled Typst)
```

No geospatial or imaging libraries are needed; images are passed to providers as
raw bytes.

---

## 3. The task, precisely

**One benchmark item consists of:**

- A single JPEG photograph of a hand (or two hands) forming one ASL
  fingerspelling sign, straight from the ASL-HG dataset.
- Ground truth: the class, one of the 36 characters 0-9 and A-Z, taken from the
  containing folder name (also encoded in the filename, but the folder is
  authoritative).

**The model's job:** return the single character the sign represents. Every
prompt (Section 6) instructs the model to end its response with exactly:

```
ANSWER: <single character>
```

**Correctness:**

- Correct if and only if the parsed character equals the ground-truth class.
  Letters are compared upper-case; digits as-is.
- A response with no parseable `ANSWER:` line (and no bare single-character
  fallback) is incorrect and is additionally tracked as a parse failure.

**Filename safety:** the dataset filename encodes the class (for example
`P1_A_5.jpg`). The model must never see it. Providers that send inline base64
(OpenAI-compatible, Anthropic) are safe by construction. The Copilot provider
attaches a file by path, so it first copies the image to a temporary file with a
neutral, randomised name.

---

## 4. Dataset

### 4.1 Source and subsetting

The raw ASL-HG dataset has 36 class folders; each holds 100 images per
participant from 10 participants (P1-P10), so 1000 images per class, ~900 MB
total. It is gitignored (`data/raw/`).

`scripts/subset_dataset.py` is a thin argparse wrapper around
`aslbench` logic that builds `data/processed/`. For every class and participant,
it keeps a random `--per-participant` sample (default 10), preserving original
filenames, so each processed class folder holds 100 images (3600 total by
default). Parameters:

| Flag | Default | Meaning |
|---|---|---|
| `--per-participant` | 10 | images kept per participant per class |
| `--seed` | none | RNG seed; when omitted, derived from the clock and recorded |
| `--force` | false | overwrite an existing `data/processed/` |

It writes `data/processed/subset_info.json` recording the seed, counts, and
parameters. The owner runs this once; make console output clear (per-class
progress and a final summary).

### 4.2 Loading and in-app subset

`aslbench.dataset` provides:

- `load_items()`: every processed image as a row (`item_id` = filename stem,
  `true_char`, `participant`, `image_number`, `filename`, `rel_path`,
  `image_abs_path`). `item_id` is unique across the dataset.
- `available_classes()`, `dataset_stats()`: for the sidebar summary.
- `build_subset(n_per_class, seed)`: for every class, sample `n_per_class`
  images without replacement (no duplicates) using a recorded seed, then
  concatenate. All 36 classes are always included, so the subset has
  `n_per_class * n_classes` rows. This is the only subsetting the app performs;
  the total image count shown in the UI is `n_per_class * 36`.

There is a single fixed dataset (`data/processed/`); the app has no dataset
picker.

---

## 5. Providers and models

### 5.1 providers.yaml

Providers are declared statically; models are never declared statically. The app
enumerates models live from whichever provider is selected.

```yaml
providers:
  - id: copilot
    type: copilot_sdk
    label: "GitHub Copilot"
  - id: anthropic
    type: anthropic
    label: "Anthropic API"
    api_key_env: ANTHROPIC_API_KEY
  - id: openai
    type: openai_compatible
    label: "OpenAI API"
    base_url: "https://api.openai.com/v1"
    api_key_env: OPENAI_API_KEY
  - id: lmstudio
    type: openai_compatible
    label: "LM Studio (local)"
    base_url: "http://localhost:1234/v1"
    api_key_env: null
  - id: omlx
    type: openai_compatible
    label: "oMLX (local)"
    base_url: "http://127.0.0.1:8000/v1"
    api_key_env: null
```

A provider is configured when its credential resolves. The app's provider picker
lists only configured providers.

### 5.2 Provider interface

```python
@dataclass
class ModelInfo:
    id: str
    label: str
    vision: bool | None   # None = capability unknown

@dataclass
class CompletionResult:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    error: str | None

class Provider(Protocol):
    id: str
    label: str
    def is_configured(self) -> bool: ...
    def list_models(self) -> list[ModelInfo]: ...
    def complete(self, model: str, prompt: str, image_path: Path) -> CompletionResult: ...
```

All three implementations are synchronous to the caller. Wrap each `complete`
with up to 2 retries and exponential backoff (2 s, 8 s); on final failure return
a `CompletionResult` with an error sentinel text so the runner records a failed
item rather than crashing. Providers must be safe to call from multiple runner
threads at once.

### 5.3 Implementations

- **openai_compat.py**: covers OpenAI, LM Studio, and oMLX server exposing an
  OpenAI endpoint. `list_models` maps to `ModelInfo(vision=None)`. `complete`
  sends one user message with a text part and an inline `image_url` data URL
  (`data:image/jpeg;base64,...`), `max_tokens` 2000, temperature 0 where
  accepted.
- **anthropic_provider.py**: `list_models` sets `vision=True`. `complete` sends
  a base64 `image` block (`media_type: image/jpeg`) plus a text block.
- **copilot_provider.py**: owns a dedicated asyncio loop on a background thread.
  `list_models` filters to vision-capable models. `complete` copies the image to
  a neutral temp filename, creates a per-item session with `available_tools=[]`
  (single-shot, no agentic loop), attaches the neutral file, collects the
  response, destroys the session, and deletes the temp copy. Consult
  `copilot-sdk-docs/features/image-input.md` and `agent-loop.md`.

---

## 6. Prompt templates

Static Markdown files in `src/aslbench/prompts/`; `render_prompt(template_id)`
returns the body (no per-item placeholders). Each ends by demanding the exact
final line `ANSWER: <single character>`. The app lets the user pick the
template; the id is recorded in run config.

- **v1_zeroshot.md** (minimal): states the task and the valid output space; no
  class list, no extra guidance.
- **v2_class_list.md**: lists all 36 classes explicitly, flags the classic
  confusions, and distinguishes the letter "O" (one hand, rounded O) from the
  digit "0" (two hands forming a closed ring).
- **v3_reasoning.md**: v2 plus explicit numbered reasoning steps (describe the
  hand, count hands, list and rule out candidates, watch the confusions, commit).

---

## 7. Runner

`aslbench/runner.py`. A run evaluates one subset with one prompt template against
one or more models; every model sees the identical item list.

- `start_run(model_specs, template_id, n_per_class, run_note="") -> run_slug`:
  - `model_specs`: non-empty list of `{provider_id, model_id}`. Duplicate
    provider+model pairs are rejected.
  - Samples the subset via `build_subset(n_per_class, seed)` with a freshly
    generated, recorded seed. Records the sampled item ids and seed in
    `config.json`; every model thread iterates exactly this list.
  - Validates inputs, creates the run folder, spawns one thread per model,
    returns immediately.
- Run slug and folder name: `aslbench-{YYYYMMDD-HHMMSS}`. Models, providers, and
  counts live in `config.json`, not the filename.
- One `threading.Thread` per model, each iterating the shared item list
  sequentially. Threads share no mutable state; each writes only inside its own
  subfolder.
- Folder contents:
  - `config.json`: `run_slug`, `template_id`, `item_ids`, `sample_seed`,
    `n_per_class`, `n_items`, `n_classes`, `run_note`, `started_at`,
    `package_version`, `providers_hash`, and `models` (each with `provider_id`,
    `provider_label`, `model_id`, `model_label`, `model_slug`).
  - `state.json` (root): `{status, started_at, finished_at, models: {slug:
    status}}`, atomic replace under a lock; overall status becomes `done` when
    every model finishes (`error` if any errored).
  - `models/<slug>/state.json`: `{status, completed, total, current_item_id,
    started_at, finished_at, error}`, atomically replaced after every item.
  - `models/<slug>/items.jsonl` (per item) and `items.parquet` (at end): per
    item `item_id`, `true_char`, `participant`, `raw_response`,
    `predicted_char`, `parse_ok`, `correct`, `latency_s`, `input_tokens`,
    `output_tokens`, `provider_error`.
  - `models/<slug>/summary.json`: the Section 8 per-model metrics.
- One active run at a time via a module handle plus a `runs/ACTIVE` lock file
  (cleared on app start). A provider failure after retries scores that item
  incorrect with `provider_error` set; an unhandled exception in one model's
  thread marks only that model `error`.
- `list_runs()` and `load_run(run_slug)` back the History, Results, and export.

---

## 8. Scoring and reporting

`aslbench/scoring.py`. All metric functions are pure (DataFrame in, dict/
DataFrame out) and unit-tested.

**Parsing:** prefer the last `ANSWER: X` line (regex allowing an optional
surrounding backtick/quote); fall back to a bare single-character reply. Letters
are upper-cased. Anything outside the 36 classes is a parse failure.

**Per-model metrics** (stored in `summary.json`):

- `accuracy` (headline), with a 95% percentile bootstrap CI (1000 resamples,
  fixed seed).
- `macro_precision`, `macro_recall`, `macro_f1` (macro-averaged over the classes
  present; parse failures count as an incorrect prediction).
- `parse_failure_rate`, `provider_error_rate`.
- `per_class_table`: accuracy, precision, recall, F1, and support per class.
- `per_participant_table`: accuracy stratified by participant.
- `confusion_long` / `confusion_heatmaps`: the required confusion matrix (true
  vs predicted), with a bucket for parse failures.
- `most_confused`: top off-diagonal (true != predicted) pairs.

**Cross-model comparison** (`compare_models`, since all models scored identical
items):

- `comparison_table`: metrics as rows, models as columns.
- `outcome_matrix`: items as rows, `true_char` plus each model's outcome
  (correct / incorrect / parse-failure).
- `pairwise_agreement`: McNemar-style discordance counts per model pair.
- `hardest_easiest`: items all models missed / all got; `hardest_classes`:
  mean per-class accuracy across models.

Every visualization must render for 1 to at least 6 models, with a consistent
per-model color reused across all figures.

---

## 9. Dash application

Entry point: `python -m aslbench.app`. Use `dash-bootstrap-components`. Single
user, local, no auth.

### 9.1 Layout

**Sidebar (top to bottom):**

1. **Dataset summary**: a one-line description of `data/processed/` (class,
   participant, and image counts), or a hint to run the subset script if it is
   missing.
2. **Images per class** (`dcc.Dropdown`, options blank + 1-10): no default. Below
   it a live `Total images: N x 36` readout. Until a number is chosen, the
   template picker, "+ Add model" button, and Run button stay disabled.
3. **Prompt template picker** with a collapsible rendered preview.
4. **Model sections**: a dynamic list of "model cards" plus "+ Add model". Each
   card has a provider picker, a model picker populated live from
   `provider.list_models()` (with a spinner and inline error), and a remove
   button. Use pattern-matching callbacks (`{"type": ..., "index": n}` with
   `MATCH`/`ALL`). Duplicate provider+model pairs are rejected with a warning.
   Start with zero cards.
5. **Run button**: disabled until a subset size is chosen, a template is chosen,
   at least one complete model card exists, no card is half-complete, and no run
   is active.

**Main area, tabs:**

- **Run**: one labeled progress bar per model plus an overall bar, driven by a
  1 s `dcc.Interval` reading the atomic state files; a live table of recent
  completions (model, item id, true, predicted, correct); a "View results"
  button when done.
- **Results** (comparison-first, any number of models): run selector defaulting
  to the latest finished run; the metrics comparison table; the accuracy/macro
  F1 grouped bars with CIs; per-class accuracy bars; confusion-matrix small
  multiples; per-participant bars; most-confused tables; the pairwise agreement
  heatmap (2+ models); and an item explorer whose rows open the image beside
  each model's raw response.
- **History**: table of all runs (slug, images per class, items, models,
  per-model accuracy, status, date); selecting one loads it into Results.
- **Export**: pick a run and PDF or HTML; show the render log on failure and the
  output path on success.

Model colors are assigned once per run and reused everywhere.

### 9.2 Dash red flags, addressed

- Long job vs blocking callbacks: the run executes in the runner's background
  threads; the Run callback only calls `start_run`. Progress arrives via
  `dcc.Interval` polling atomic `state.json`. `DiskcacheManager` is the noted
  upgrade path for multi-user hosting.
- Browser refresh: all run state lives on disk; the active run is re-detected
  from `runs/ACTIVE` on load.
- Payload bloat: never put per-item results or images into `dcc.Store`;
  callbacks read run folders server-side. Images are streamed by a small Flask
  route (`/image/<item_id>`) that resolves the item id to a file under
  `data/processed/`, guarded against path traversal.

---

## 10. Quarto export

The run folder is the intermediate; `report/report.qmd` reads it via the
`run_dir` parameter at render time.

Front matter:

```yaml
format:
  html:
    embed-resources: true
    toc: true
  typst:
    toc: true
params:
  run_dir: ""
```

`embed-resources: true` yields a single self-contained HTML. The body (Python
cells run with the conda env kernel) loads `config.json`, the per-model frames
and summaries via `aslbench.runner`/`scoring`, and renders, in order: a run
metadata block and model table; the comparison section (comparison table,
accuracy/macro F1 bars, per-class bars, confusion matrices, per-participant
bars, pairwise agreement); then per-model sections (per-class table,
most-confused pairs, a sample-misclassifications table). Keep all figure/table
inputs format-agnostic so the same template renders to both HTML and PDF; do not
reference image files outside the report directory (Typst rejects them).

`aslbench/export.py`:

```python
def export_run(run_slug: str, fmt: Literal["pdf", "html"]) -> Path:
    to = "typst" if fmt == "pdf" else "html"
    subprocess.run(["quarto", "render", "report/report.qmd", "--to", to,
                    "-P", f"run_dir:{run_dir.resolve()}", "--output", out_name],
                   check=True, capture_output=True, text=True)
    # move the output into exports/
```

PDF uses Quarto's bundled Typst engine (no LaTeX). If Typst is missing, surface a
clear message about `quarto install tinytex`.

---

## 11. Tests

Use pytest; no network (mock providers).

- `test_parsing.py`: `ANSWER:` extraction (last match wins, upper-casing,
  backticks, bare single char, invalid char, missing line, empty).
- `test_scoring.py`: `score_item` exact match; summary accuracy, parse-failure
  rate, per-class and per-participant tables; confusion buckets parse failures;
  bootstrap determinism.
- `test_dataset.py`: filename parsing; `load_items` on a synthetic processed
  tree; canonical class order; `build_subset` size, no duplicates, determinism,
  all classes included.
- `test_prompts.py`: every template ends with the `ANSWER:` contract; v2/v3 list
  the classes and the O/0 caveat.
- `test_runner.py`: with `FakeProvider`s, a two-model run produces coherent root
  and per-model state, both models receive the identical item list, an exception
  in one model's thread marks only it errored, duplicate pairs are rejected.
- `test_comparison.py`: `compare_models` comparison table, outcome matrix,
  pairwise agreement, hardest/easiest, hardest classes, and single-model
  degradation.

---

## 12. Implementation phases

1. Skeleton: `environment.yml`, `pyproject.toml`, package layout,
   `providers.yaml`, config with the 36 classes.
2. Dataset module plus `scripts/subset_dataset.py`; build `data/processed/` once.
3. Providers: protocol plus all three implementations plus live model
   enumeration; verify the filename never leaks.
4. Runner and scoring with tests passing, including comparison functions.
5. Dash app: layout with the images-per-class gate and dynamic model cards,
   progress polling, comparison results, history, image route.
6. Quarto export: report.qmd (comparison plus per-model sections), export.py,
   both formats verified.
7. Smoke validation: a small real run through the UI against at least two models
   (a local model plus an API model where keys exist); export both formats.

Out of scope: deployment, the written assessment prose, audio/video, any
LLM-as-judge.

---

## 13. Known uncertainties for the implementer

- Copilot SDK Python method casing and session teardown API: follow
  `copilot-sdk-docs/` as ground truth; the filename-neutralising copy is
  mandatory there.
- Local (LM Studio / oMLX) endpoints vary in whether they accept `temperature`
  and how they report vision capability; tolerate both.
- Very large per-class subsets multiply request counts by 36; the app caps the
  dropdown at 10 per class (360 images) to keep runs tractable.
