# aslbench

aslbench is a benchmark for fine-grained visual perception in frontier vision
language models (VLMs). A model sees one photograph of an American Sign Language
(ASL) fingerspelling handshape and must identify one of 36 characters: digits
0 to 9 or letters A to Z. A prediction is correct only when it exactly matches
the ground-truth folder label. No human or model judge participates in scoring.

The project includes a local Dash application for running and inspecting model
comparisons, archived runs, a parameterized Quarto run report, and a final
assessment report based on the archived final run.

## Why this dataset

The benchmark uses **ASL-HG** (Pranto et al., *Data in Brief*, January 2026,
DOI [10.1016/j.dib.2026.112492](https://doi.org/10.1016/j.dib.2026.112492)),
available from [Mendeley Data](https://data.mendeley.com/datasets/j4y5w2c8w9/1).

It is a good fit because:

* It is a hard task for VLMs: fingerspelling depends on precise finger, thumb,
  palm, and hand-count information.
* It has 36 classes, digits 0 to 9 and letters A to Z, with 100 source images
  per class and participant.
* It contains photographs of 10 volunteers in Dhaka, Bangladesh, captured in
  natural indoor and outdoor settings.
* It distinguishes the one-handed letter `O` from the two-handed digit `0`.
* It was published in January 2026, which reduces but does not eliminate the
  risk of training-data contamination.

## Quick start

You need a clone of this repository, a terminal opened in its root directory,
and Python 3.11 or newer. Check with `python3 --version` on macOS or Linux, or
`py --version` on Windows. You also need one configured model provider before
you can run a benchmark. Quarto is required only to export reports; see
[Quarto's installation guide](https://quarto.org/docs/get-started/) and confirm
that `quarto --version` works in a new terminal after installation.

Choose one Python installation path. Conda is recommended because it isolates
this project's packages from other Python work.

### Option 1: Conda environment

Conda creates an isolated Python environment. Install a Conda distribution such
as [Miniforge](https://github.com/conda-forge/miniforge) or Miniconda first if
the `conda` command is unavailable.

```bash
conda env create -f environment.yml
conda activate aslbench
python -m pip install -e .
```

Run `conda activate aslbench` in each new terminal before using the app, tests,
or reports.

### Option 2: Existing base Python

This path installs the project and its provider, test, and report dependencies
into the Python installation you already use. It is convenient for a quick local
evaluation, but those packages will share that Python installation with your
other work.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[providers,test,report]"
```

On Windows, replace `python3` with `py -3.11` or the Python launcher installed
on your machine. Quarto remains a separate system installation in both paths.

## Dataset

The repository includes the processed benchmark subset in `data/processed/`, so
you can run the app immediately. It contains
3,600 JPEGs: 10 images for every combination of 36 classes and 10 participants.
The raw ASL-HG dataset is about 900 MB and is not stored in git.

### Layout

Both the raw and processed datasets use one folder per class:

```
data/raw/asl_hg_dataset/<class>/P<participant>_<class>_<image>.jpg
data/processed/<class>/P<participant>_<class>_<image>.jpg
```

A filename such as `P1_A_5.jpg` means participant 1, class A, image 5. The
source filename encodes the class, but the benchmark never sends it to a model.

### Rebuild the processed subset

To reproduce the processed dataset from the raw source, download and unpack the
ASL-HG files so their class folders are directly inside
`data/raw/asl_hg_dataset/`. Then run one of the following commands:

```bash
python scripts/subset_dataset.py
python scripts/subset_dataset.py --per-participant 5 --seed 42
python scripts/subset_dataset.py --force
```

The default command retains 10 seeded images per participant and class. The
script records its seed and resulting counts in `data/processed/subset_info.json`.
Use `--force` only when you intend to replace the existing processed dataset.

## Configure a model provider

Provider definitions live in `providers.yaml`. The app lists models live from a
selected provider. Configure at least one of the following before starting a
benchmark:

| Provider | What you need |
|---|---|
| GitHub Copilot | Install and authenticate the GitHub Copilot CLI, then verify `copilot --version`. The app lists available vision models after you select Copilot. |
| Anthropic API | Set `ANTHROPIC_API_KEY` in the terminal where you start the app. |
| OpenAI API | Set `OPENAI_API_KEY` in the terminal where you start the app. |
| LM Studio | Start an OpenAI-compatible local server with a vision-capable model at `http://localhost:1234/v1`. |
| oMLX | Start an OpenAI-compatible local server with a vision-capable model at `http://127.0.0.1:8000/v1`. |

For example, on macOS or Linux:

```bash
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

On Windows PowerShell, use `$env:ANTHROPIC_API_KEY = "your-key"` or
`$env:OPENAI_API_KEY = "your-key"`. Do not place API keys in `providers.yaml`
or commit them to the repository. For a different local endpoint, edit the
corresponding `base_url` in `providers.yaml`, then restart the app.

The benchmark sends image pixels to the configured provider. Cloud-compatible
providers receive base64 image data, and the Copilot provider attaches a
temporary image with a neutral name. The class-encoding filename is never sent.

## Run the app

Start the application from the repository root:

```bash
python -m aslbench.app
```

Open the local URL printed in the terminal, normally `http://127.0.0.1:8050/`.
Keep the terminal running while you use the app. If the app reports that the
processed dataset is missing, follow the dataset rebuild instructions above. If
the model dropdown cannot list models, check the provider credential or local
server in the same terminal where the app is running.

To start a benchmark:

1. Select **Images per class**. Every run includes all 36 classes, so for example, selecting 3 creates a 108-image run for each selected model.
2. Select a **Prompt template**. Use the preview control to inspect its exact
   wording.
3. Select **Add model**, choose a provider, then choose one vision-capable
   model. Add more cards for a paired comparison.
4. Optionally enter a run note, then select **Run benchmark**.
5. Monitor the per-model and total progress bars in the **Run** tab. Use
   **Stop run** to stop after each in-flight request finishes.

Every model in a run receives the same sampled images in the same order. The
run folder records the sample seed, item identifiers, prompt, models, responses,
and scores, so results can be reproduced or inspected later.

## View a saved run in Results

When a new run finishes, select **View results** in the completion message to
open it directly. To load any single archived run instead:

1. Open the **Results** tab.
2. Choose the desired run slug from the **Run** dropdown at the top, for example
   `aslbench-20260713-203909 [done]`.
3. The comparison table, charts, confusion matrices, paired McNemar tests,
   confusion tables, and item explorer load for that run.
4. In the item explorer, select a row to view the evaluated image and each
   model's response.

The **History** tab lists runs and permits deletion. It does not load a run into
Results, so use the Results-tab dropdown for analysis.

## Prompt templates

All templates require the final line `ANSWER: <single character>`.

* `v1_zeroshot` gives the minimal task instruction and no class list.
* `v2_class_list` lists the 36 valid classes and explains the `O` versus `0`
  distinction.
* `v3_reasoning` adds an explicit, structured visual-analysis procedure.

## Metrics and analysis

The dashboard and run report show:

* overall accuracy and macro F1 with deterministic 95% bootstrap intervals;
* a cross-model table with accuracy, macro F1, parse-failure rate,
  provider-error rate, and a random-choice baseline;
* per-class accuracy-difference charts when two or more models are compared;
* row-normalized confusion matrices, including a parse-failure bucket;
* exact paired McNemar tests with Holm correction when two or more models are
  compared;
* the most frequent true-to-predicted confusions for each model;
* an item explorer with the image, prediction status, optional thinking text,
  and raw response for every selected item.

Correctness is exact match. A missing or invalid final answer is incorrect and
tracked as a parse failure. A provider timeout or failure is also incorrect and
tracked separately.

## Export a run report

Quarto must be installed and available on your `PATH`. In the app, open the
**Export** tab, choose a completed run and HTML or PDF, then select **Export**.
Rendered reports are written to `exports/`.

You can also export from Python:

```python
from aslbench.export import export_run

export_run("aslbench-20260713-203909", "html")
export_run("aslbench-20260713-203909", "pdf")
```

HTML reports are self-contained. PDF export uses Quarto's bundled Typst engine,
so a separate LaTeX installation is not required.

## Generate the final report

The final assessment report is `final-report.qmd`. It uses the archived final
run, `aslbench-20260713-203909`, and appends its detailed run analysis. Quarto,
the report dependencies, and the archived run folder are required.

```bash
quarto render final-report.qmd --to html -P output_fmt:html
quarto render final-report.qmd --to typst -P output_fmt:typst
```

The commands create `final-report.html` and `final-report.pdf` beside the QMD.

## Run tests

```bash
python -m pytest
```
