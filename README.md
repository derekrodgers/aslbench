# aslbench

A novel benchmark measuring how well frontier vision language models (VLMs) can
recognise American Sign Language (ASL) fingerspelling handshapes. A model is
shown a single photograph of a hand (or two hands) forming one fingerspelling
sign and must name the character it represents: a digit 0-9 or an uppercase
letter A-Z. Answers are scored against the ground-truth class (the folder the
image came from), so scoring is fully objective, categorical, and needs no human
or LLM judge.

The task is objectively scored (exact-match against a fixed 36-way label),
uses a public credential-free dataset, and probes fine-grained visual perception
of fingers and gestures, an area where general-purpose VLMs tend to lag the
specialised CNNs usually trained on this kind of data.

## Why this dataset

The benchmark uses **ASL-HG** (Pranto et al., *Data in Brief*, January 2026,
DOI [10.1016/j.dib.2026.112492](https://doi.org/10.1016/j.dib.2026.112492)),
available on Mendeley Data at
<https://data.mendeley.com/datasets/j4y5w2c8w9/1>.

It is a good fit for this benchmark because:

- It is a **hard task for VLMs**: models frequently struggle with the precise
  configuration of fingers and gestures, exactly what fingerspelling recognition
  demands.
- It has **36 classes** (0-9 and A-Z) with 100 samples per class per participant
  in the raw dataset. This repository ships a much smaller subset under
  `data/processed/` (see below).
- The images come from **10 volunteers in Dhaka, Bangladesh**, captured as real
  smartphone HD photos with natural indoor and outdoor lighting and backgrounds,
  not sterile studio captures.
- It is **balanced** across subjects, genders, and skin tones.
- It carefully **distinguishes the letter "O" from the digit "0"** (the latter
  signed as a two-handed zero), a distinction most datasets skip.
- It is **recent** (January 2026), which keeps the risk of training-data leakage
  low.

## Setup

Environment management uses conda.

```bash
conda env create -f environment.yml
conda activate aslbench
pip install -e .
copilot --version   # verify the GitHub Copilot CLI is available (if using the Copilot provider)
quarto --version    # verify Quarto >= 1.4 (PDF export uses its bundled Typst)
```

## The dataset

The raw ASL-HG dataset is large (~900 MB) and is **not tracked in git**
(`data/raw/` is gitignored). The benchmark runs against a smaller, reproducible
subset in `data/processed/`, produced by `scripts/subset_dataset.py`.

### Layout

Both the raw and processed datasets use one folder per class:

```
data/raw/asl_hg_dataset/<class>/P<participant>_<class>_<image>.jpg
data/processed/<class>/P<participant>_<class>_<image>.jpg
```

A filename such as `P1_A_5.jpg` means "participant 1, class A, image 5". The
original filenames are preserved in the subset.

### Building the subset

For every class and every participant, the subsetting script keeps a random
sample of `--per-participant` images (default 10), so each class folder in
`data/processed/` holds 10 participants x 10 images = 100 images (3600 images
total by default). Sampling is seeded and the seed is recorded in
`data/processed/subset_info.json`.

```bash
python scripts/subset_dataset.py                    # default: 10 per participant
python scripts/subset_dataset.py --per-participant 5 --seed 42
python scripts/subset_dataset.py --force            # overwrite an existing subset
```

### Reproducing from scratch

To rebuild `data/processed/` yourself for a reproducibility check, download the
raw dataset from <https://data.mendeley.com/datasets/j4y5w2c8w9/1>, unpack it to
`data/raw/asl_hg_dataset/` (so the class folders sit directly inside), then run
the subsetting script above.

## Providers and credentials

Providers are declared in `providers.yaml`; models are enumerated live from each
provider. A provider is available when its credential resolves:

- `anthropic`: set `ANTHROPIC_API_KEY`
- `openai`: set `OPENAI_API_KEY`
- `lmstudio`: a local OpenAI-compatible server at `http://localhost:1234/v1`
  (for example a Qwen VL model served through LM Studio or MLX)
- `copilot`: the GitHub Copilot CLI must be authenticated

Images are always sent to models as raw pixels (inline base64, or, for the
Copilot provider, a copy with a neutralised filename). The class-encoding
filename is never revealed to the model.

## Running the app

```bash
python -m aslbench.app
```

Then open the printed URL. In the sidebar:

1. Choose **images per class** (1-10) from the dropdown. There is no default; the
   template, model, and run controls stay disabled until you pick a number. The
   sidebar shows the resulting total (for example 3 per class gives
   `Total images: 108` since 3 x 36 = 108).
2. Pick a **prompt template**.
3. Add one or more **model cards** (each is a provider plus a live-enumerated
   model).
4. Start the run.

Every model sees the identical sampled images (all 36 classes are always
included, minimum one image per class), so results are directly comparable.
Progress, comparison results, history, and export are on the main tabs.

## Prompt templates

Three templates are provided so prompt wording can be iterated after seeing
results. All three instruct the model to end its reply with a single line
`ANSWER: <single character>`.

- `v1_zeroshot`: minimal instruction, no class list.
- `v2_class_list`: lists all 36 valid classes and warns about the letter "O"
  versus the two-handed digit "0".
- `v3_reasoning`: adds explicit "describe the hand, then decide" reasoning steps.

## Statistics and visualizations

Because the labels are categorical, the dashboard and the exported report show:

- overall accuracy (with a 95% bootstrap confidence interval) and macro F1;
- a metrics comparison table across models (accuracy, macro precision/recall/F1,
  parse-failure and provider-error rates);
- per-class accuracy bars;
- a confusion matrix per model (true vs predicted, with a bucket for unparseable
  answers);
- per-participant accuracy bars;
- the most-confused class pairs per model;
- pairwise both-correct agreement across models (when two or more are run);
- an item explorer: click any item to see the image alongside each model's full
  raw response.

## Exporting reports

The exported report contains every statistic and visualization shown on the
dashboard. From the Export tab, or programmatically:

```python
from aslbench.export import export_run
export_run("<run_slug>", "html")  # or "pdf"
```

PDF export uses Quarto's bundled Typst engine (no LaTeX install needed).

## Tests

```bash
pytest
```

## Notes

- The model is asked to return a **single character** (for example `A` or `0`);
  a reply with no parseable answer is scored incorrect and tracked as a parse
  failure.
- Correctness is exact match against the ground-truth class. There is no partial
  credit.
- The class-encoding filename is never sent to the model, so the answer cannot
  leak through the attachment name.

## Generating the final report

The final assessment report is `final-report.qmd` in the repository root. It
uses the archived `aslbench-20260713-203909` run for its results and detailed
appendix. Activate the project environment, then render it with Quarto:

```bash
conda activate aslbench
quarto render final-report.qmd
```

The QMD defines both HTML and Typst formats, so this creates
`final-report.html` and `final-report.pdf` alongside the source file. HTML is
self-contained. PDF rendering also requires the `kaleido` package included by
the project report dependencies.
