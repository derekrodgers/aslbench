# Benchmark design brief for Fable

I am completing a take home technical assessment as part of a job application
process. The assessment, which I have pasted in full below, asks me to design,
build, and present a novel benchmark that measures something meaningful about
frontier AI model capabilities. After I submit it, I will present it live to the
team and answer questions about my design choices.

This brief captures the concept I settled on and the constraints it satisfies,
so that a coding agent can build it from a single plan document (PLAN.md) without
needing the conversation that produced it. The chosen benchmark is **aslbench**:
36-way recognition of American Sign Language (ASL) fingerspelling handshapes by
frontier vision language models.

## The assessment

This is the take home assessment I am working from, reproduced exactly as given
to me:

```
Take-Home Assessment: Novel LLM Benchmark

Overview

Design and implement a novel benchmark that meaningfully differentiates the capabilities of current frontier AI models. This may target language, reasoning, multimodal, or any other capability domain.

Requirements

Your benchmark must be:
- Novel. Not a wrapper around an existing evaluation suite or a variation on a known benchmark. Test a capability, failure mode, or reasoning pattern of your choosing.
- Non-saturated. At least some current models should score meaningfully below perfect, and at least some should score meaningfully above zero.
- Reproducible. Another person should be able to run your benchmark and obtain comparable results.
- Quantitatively scored. Results must be measurable, not purely subjective.

Deliverables

- The benchmark: code, prompts, interactive application, or whatever form it takes.
- A written report covering:
  - What your benchmark measures and why it matters
  - Methodology and scoring design
  - Results across a minimum of 2 models
  - Analysis of what the results reveal about model capabilities
  - Limitations and what you would improve with more time
- A git repository with your development history.
- Bonus: An interactive benchmark. Something we can run, play, or experience ourselves.

Tools and Access

Use whatever tools and models you have available. There are no required providers or frameworks. You should use AI tools to assist your build.

Next Steps

Following your submission, you will present your benchmark in a conversation with the team. Be prepared to walk through your design rationale, discuss your results, and answer questions about your methodology.

```

## The chosen benchmark

**aslbench** will show a vision language model a single photograph of a hand (or two
hands) forming one ASL fingerspelling sign and asks it to name the character: a
digit 0-9 or an uppercase letter A-Z (36 classes). The prediction is scored by
exact match against the ground-truth class. It is a pure fine-grained perception
task: getting it right means resolving the precise configuration of fingers,
thumb, and palm, which is exactly the kind of gesture reading that general
purpose VLMs tend to handle worse than the specialised CNNs usually trained on
this data.

The dataset is **ASL-HG** (Pranto et al., *Data in Brief*, January 2026, DOI
10.1016/j.dib.2026.112492), free and credential-free on Mendeley Data at
https://data.mendeley.com/datasets/j4y5w2c8w9/1. It has 36 classes with 100
samples per class per participant from 10 volunteers in Dhaka, Bangladesh, shot
as real smartphone HD photos under natural lighting. It carefully distinguishes the letter "O" from the two-handed digit "0", and is recent enough to keep leakage risk low.

## Benchmark idea constraints

These are the constraints the concept was chosen to satisfy. Four are hard constraints; one is a strong preference; one is an open degree of freedom.

Hard constraints. All four hold for aslbench.

1. Objectively scored. A fixed ground truth label, categorical, determines
   whether an answer is correct. No human judge and no LLM judge anywhere in the
   scoring path. aslbench compares the predicted character to the folder label.
2. Public dataset. A free, publicly accessible dataset with unambiguous labels
   already exists, from an established academic source, needing no paid access,
   institutional credentials, or from-scratch collection. ASL-HG on Mendeley
   Data satisfies this.
3. Genuinely novel. Ideally no one has benchmarked any model on the exact task;
   at minimum, no frontier LLM or VLM has. ASL fingerspelling recognition has
   been tackled with specialised CNNs, but not, to my knowledge, with general
   purpose frontier VLMs on this dataset, which is the gap we care about.
4. Meets the assessment's own bar. Not saturated and reproducible: fingerspelling
   is hard enough that VLM scores land below perfect while staying above zero,
   and the seeded subset plus archived runs make results reproducible.

Strong preference (satisfied here in spirit):

5. Favor tasks that stress a capability where general purpose frontier models
   are known to be weaker than narrow specialised models, so the benchmark
   reveals a real gap rather than re-measuring a solved skill. Fine-grained
   perception of hands and gestures is such a capability: it demands careful
   discrimination of near-identical handshapes (A vs S vs T, M vs N, O vs 0)
   rather than retrieval of a memorized fact.

## How to interact with me

Whenever you need my input, whether to choose between two reasonable approaches
or to clarify a constraint, use whatever interactive prompt, dialog, or multiple
choice mechanism is available to you rather than asking an open ended question in
plain text. I would rather tap or click a choice than type a paragraph.

## Application and architecture requirements

Everything in this section is context for PLAN.md.

### The interactive application is a core design consideration, not an afterthought

The bonus goal, an interactive benchmark the team can run and experience, should
shape the architecture from the start. Every piece of benchmark logic (dataset
loading, model invocation, scoring) must be written so the application calls it
directly, rather than existing only as a separate command line tool the
application wraps.

Concretely, a person using the application needs to be able to do the following.

1. Pick which model to run the benchmark against, from any of the backends
   described below. Multiple models can be selected for a single head-to-head
   run.
2. Pick a prompt template, since I expect to iterate on prompt wording after
   seeing results.
3. Choose how much of the dataset to run against. There is a single fixed
   dataset (`data/processed/`), so the only subset control is how many images to
   sample per class, chosen from a dropdown offering 1 through 10. All 36 classes
   are always included (minimum one image per class), so the total sent to the
   model is that number times 36; the UI shows the running total. The default
   selection is empty, and it is not a valid value: the template, model, and run
   controls stay disabled until a number is chosen, so a run only happens once
   someone has deliberately chosen something. Sampling is without replacement
   (no duplicate images).
4. Watch a progress indicator while a run is in progress, since a full run may
   take a while.
5. See the resulting statistics once a run finishes, including a confusion
   matrix.

On those statistics, research what standard reporting looks like for an
objectively scored, ground truth labeled multi-class recognition benchmark
(accuracy with confidence intervals, per-class and macro precision/recall/F1, a
confusion matrix, per-subject stratification, cross-model comparison) and propose
that reporting in the plan.

I am very familiar with Plotly Dash and would default to it, but I want
independent reasoning here. If Dash is a reasonable choice, address the red flags
that come from pairing a long running benchmark job with a statistics dashboard
in Dash.

The model must be asked to return a single character (for example `A` or `0`).
The image filename encodes the class, so it must never be sent to the model.

### Exporting results

Results need to be exportable as a PDF or an HTML-embed document (user chooses). I
have Quarto installed. The rough idea: save a run out to an intermediate on disk,
then have a parameterized QMD template read it at render time. Work out the
specifics: what the intermediate contains, how the QMD receives its parameters,
and how rendering is triggered from inside the application. The report must
contain every statistic and visualization shown on the dashboard.

### Environment management

Conda is already installed and environment management for this project must use
conda, not venv and not virtualenv. The plan must create a dedicated conda
environment and install every dependency into it. This is a requirement.

### Running the benchmark against different backends

I want to run the benchmark against a model reached through any of the following,
and the plan should explain how each is wired in.

1. My GitHub Copilot subscription, through whichever mechanism is genuinely the
   most sensible way to reach it programmatically. If the Copilot SDK is the best
   fit, its documentation is in `copilot-sdk-docs/`; read from there rather than
   the web.
2. A direct Anthropic or OpenAI subscription or API key.
3. A model running locally, exposed through an OpenAI compatible or Anthropic
   compatible endpoint. I run local models through MLX and LM Studio and want to
   point the benchmark at something like a Qwen VL model running locally.

Design the model calling layer so that swapping between these three is a matter of
configuration, not a rewrite.

### Deployment

Fly.io is the likely eventual deployment target, but treat it purely as a stretch
goal. Do not let it drive any core design decision, and do not include deployment
steps in PLAN.md; we will handle deployment separately once the benchmark works.

## What to do

Write a single file named PLAN.md containing the full architecture and
implementation plan for the aslbench benchmark described above, incorporating
everything in the application and architecture requirements section.
