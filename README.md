# RouteGuard AI

### Benchmark-Driven LLM Routing & Quality Evaluation Platform

RouteGuard AI is an intelligent multi-model inference platform that dynamically selects the **most cost-efficient LLM capable of meeting a configurable quality requirement**.

Instead of routing requests using hardcoded model tiers such as *small / medium / large* or assuming that a more expensive model is always better, RouteGuard learns task-specific model performance from empirical benchmarks.

For every request, RouteGuard asks:

> **What is the least expensive model that has demonstrated sufficient quality for this type of task?**

The system combines task-aware routing, empirical benchmarking, statistical confidence bounds, deterministic evaluation, independent LLM-as-a-Judge evaluation, cost-aware fallback, and explainable routing decisions.

---

## Why RouteGuard?

Production LLM systems often face a tradeoff:

- stronger models may improve quality but cost more;
- cheaper models reduce inference cost but may fail on harder tasks;
- static routing rules become stale as models change;
- raw benchmark accuracy ignores inference cost;
- routing solely by price can degrade quality;
- routing everything to the strongest model wastes resources.

RouteGuard treats model selection as a **quality-constrained optimization problem**.

Conceptually:

```text
minimize    Expected Cost per Successful Response

subject to  Estimated Quality >= Required Quality
```

Rather than asking which model is "best" globally, RouteGuard asks which model is **best for the current task under the configured quality requirement**.

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │     User Prompt      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Task Classifier    │
                         │                      │
                         │ factual QA           │
                         │ explanation          │
                         │ reasoning            │
                         │ coding               │
                         │ extraction           │
                         │ summarization        │
                         │ grounded QA          │
                         │ structured output    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌────────────────────────────┐
                     │ Benchmark Profile Lookup   │
                     │                            │
                     │ model × task performance   │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Statistical Qualification  │
                     │                            │
                     │ Wilson confidence bound    │
                     │ + minimum sample evidence  │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Cost / Success Ranking     │
                     │                            │
                     │ cheapest qualifying model  │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │      Generation      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌────────────────────────────┐
                     │    Evaluation Router       │
                     └─────────────┬──────────────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
              ┌─────────────────┐     ┌─────────────────┐
              │ Deterministic   │     │ Independent     │
              │ Evaluation      │     │ LLM-as-a-Judge  │
              └────────┬────────┘     └────────┬────────┘
                       └───────────┬────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ Quality Threshold    │
                         └──────────┬───────────┘
                                    │
                           PASS ─────┴───── FAIL
                            │                 │
                            ▼                 ▼
                         Return         Next Ranked Model
                        Response              │
                                             └──► Re-evaluate
```

---

## Core Capabilities

### 1. Benchmark-Driven Model Routing

RouteGuard does not assign hand-written capability scores to models.

Offline benchmarking measures performance for each:

```text
(model, task_type)
```

pair.

The router uses this empirical history to determine which models have demonstrated sufficient quality for a particular task.

This means a low-cost model can legitimately beat a much larger model when it already satisfies the requested quality requirement.

---

### 2. Quality-Constrained Cost Optimization

For each task, RouteGuard computes:

- benchmark sample count
- observed pass rate
- Wilson lower confidence bound
- average generation cost
- average total inference + evaluation cost
- average latency
- cost per successful response

Models first have to satisfy the quality requirement.

Among qualifying models, RouteGuard prefers the lowest:

```text
Cost per Successful Response
```

This is deliberately different from simply selecting the cheapest API call.

A cheap model that frequently fails and triggers retries may ultimately be more expensive than a slightly more capable model that succeeds immediately.

---

### 3. Statistical Confidence Instead of Raw Accuracy

A model going 5/5 on a tiny benchmark should not automatically be treated as a 100%-reliable model.

RouteGuard therefore uses a **Wilson lower confidence bound** rather than raw benchmark accuracy when qualifying models.

Conceptually:

```text
Observed pass rate
        ↓
Wilson lower confidence bound
        ↓
Compare against required quality threshold
```

This makes model qualification conservative when benchmark evidence is limited.

---

### 4. Explicit Cold-Start Behavior

If RouteGuard does not have enough benchmark evidence for a task, it does **not invent a quality score**.

Instead, the request is explicitly labeled:

```text
exploration_cold_start
```

The router can select a low-cost candidate for calibration while clearly stating that the decision is not benchmark-backed.

As benchmark evidence accumulates, routing transitions from exploration to empirical quality/cost optimization.

---

### 5. Task-Aware Evaluation Router

Running another LLM to evaluate every response can defeat the purpose of cost optimization.

RouteGuard therefore decides whether an LLM judge is actually necessary.

For example:

```text
Factual QA + reference answer
        ↓
Deterministic comparison
        ↓
No judge API call required
```

while:

```text
Knowledge explanation
        ↓
Semantic quality required
        ↓
Independent LLM-as-a-Judge
```

This allows objective tasks to avoid unnecessary evaluation cost while retaining semantic evaluation for tasks where deterministic rules are insufficient.

---

### 6. Hybrid Evaluation Framework

RouteGuard supports two complementary evaluation mechanisms.

**Deterministic evaluation**

Used when correctness can be objectively checked:

- exact/reference matching
- required terms
- JSON validity
- citation format
- non-empty output

**Independent LLM-as-a-Judge**

Used for semantic criteria such as:

- instruction adherence
- completeness
- groundedness when context exists
- reference correctness when a reference exists

The judge is intentionally separated from the generation model where possible.

---

### 7. Evidence-Aware Truthfulness

RouteGuard deliberately distinguishes:

```text
A response looks good
```

from:

```text
A response has been independently verified as factually true
```

Without external evidence, truthfulness remains:

```text
UNVERIFIED
```

The LLM judge is explicitly instructed not to claim that an answer is factually correct, true, or verified when independent evidence is unavailable.

This prevents self-evaluation from being misrepresented as factual verification.

---

### 8. Quality-Triggered Model Fallback

Model selection is not the end of routing.

After generation:

```text
Generate
   ↓
Evaluate
   ↓
Quality threshold met?
   │
   ├── YES → return
   │
   └── NO
        ↓
   next ranked model
        ↓
     regenerate
        ↓
     re-evaluate
```

Fallback therefore depends on measured response quality rather than a hardcoded model escalation hierarchy.

---

### 9. Full Cost Accounting

RouteGuard tracks the cost of the **entire successful inference path**, including:

- generation tokens
- evaluation tokens
- generation cost
- judge cost
- failed attempts
- fallback attempts
- total tokens
- total end-to-end cost

The optimization target is therefore not simply first-call inference cost.

It is closer to:

```text
Total Cost Required to Produce a Quality-Validated Response
```

---

## Supported Task Categories

RouteGuard currently classifies prompts into:

| Task | Example |
|---|---|
| `factual_qa` | "What is the capital of Oman?" |
| `knowledge_explanation` | "Explain photosynthesis." |
| `reasoning` | Multi-step reasoning and comparison |
| `coding` | Generate or debug code |
| `summarization` | Summarize supplied information |
| `extraction` | Extract entities or attributes |
| `grounded_qa` | Answer using supplied context |
| `structured_output` | Produce schema/JSON-constrained output |
| `general` | Catch-all for uncategorized requests |

Task classification determines which empirical benchmark profile and evaluation strategy should be applied.

---

## Offline Benchmarking

RouteGuard includes an offline benchmarking pipeline that runs configured models against the same evaluation dataset.

```bash
python -m routeguard.benchmark --dataset data/benchmark.jsonl
```

To benchmark only selected models:

```bash
python -m routeguard.benchmark \
  --dataset data/benchmark.jsonl \
  --models gpt-4.1-nano gpt-4.1-mini
```

Benchmark execution records:

```text
model
task
benchmark case
pass / fail
generation tokens
evaluation tokens
generation cost
evaluation cost
total cost
latency
evaluation strategy
deterministic checks
judge output
```

Results are persisted in SQLite and aggregated into task-specific model profiles.

### Benchmark methodology

For each `(model, task)` pair, RouteGuard derives:

```text
sample count
observed success rate
Wilson lower confidence bound
average generation cost
average total cost
average latency
cost per successful response
```

These profiles become the empirical evidence used by the online router.

> The benchmark dataset in this repository is currently being expanded. Existing results should be treated as engineering validation rather than production-quality model performance claims until sufficient held-out coverage is available.

---

## Example Routing Decision

Consider:

```text
Explain photosynthesis.
```

RouteGuard first classifies the request:

```text
Task: knowledge_explanation
Complexity: low
```

It then retrieves empirical `knowledge_explanation` profiles for all candidate models.

Suppose several models satisfy the configured quality threshold.

RouteGuard does **not** automatically select the strongest model.

Instead:

```text
Model A → qualifies → $0.0002 / successful response
Model B → qualifies → $0.0005 / successful response
Model C → qualifies → $0.0014 / successful response
```

RouteGuard selects **Model A**.

If Model A's generated response fails online evaluation, RouteGuard automatically attempts the next ranked candidate.

---

## Candidate Models

The current configuration evaluates seven GPT model IDs:

```text
gpt-4.1-nano
gpt-4.1-mini
gpt-4.1
gpt-4o-mini
gpt-4o
gpt-5-mini
gpt-5
```

The architecture is model-registry driven; models can be added or removed through configuration rather than routing code.

Model availability, pricing, and endpoint compatibility depend on the provider/account and should be verified before benchmarking.

---

## Explainability & Observability

The Streamlit dashboard exposes the complete decision path.

For each request it can show:

```text
Task classification
Complexity
Routing mode
Selected model
Why the model was selected

Benchmark sample count
Observed pass rate
Wilson lower bound
Cost per successful response

Generation tokens
Generation cost

Evaluation strategy
Evaluation tokens
Evaluation cost

Fallback attempts
Total tokens
Total cost
Final quality status
```

The goal is to make routing decisions inspectable rather than hiding them behind a black-box model selector.

---

## API

Start the API:

```bash
uvicorn routeguard.api:app --reload
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Primary endpoints:

```text
POST /route
GET  /health
GET  /metrics
GET  /benchmarks/profiles
```

---

## Dashboard

Start Streamlit:

```bash
streamlit run dashboard.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides:

- prompt execution
- configurable quality threshold
- routing explanation
- candidate comparison
- evaluation trace
- fallback visibility
- cost/token accounting
- historical request metrics
- offline benchmark profiles

---

## Running Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy:

```text
.env.example
```

to:

```text
.env
```

For real inference:

```text
DEMO_MODE=false
OPENAI_API_KEY=<your-key>
MIN_BENCHMARK_SAMPLES=30
```

Never commit API keys.

### 4. Start the API

```bash
uvicorn routeguard.api:app --reload
```

### 5. Start the dashboard

In another terminal:

```bash
streamlit run dashboard.py
```

---

## Testing

Run:

```bash
pytest -q
```

Tests cover areas including:

- task classification
- factual QA detection
- knowledge-explanation detection
- cold-start routing
- cost-based exploration
- deterministic evaluation
- evaluation-router behavior
- judge skipping for objectively validated answers
- structured-output evaluation
- Wilson confidence bounds
- benchmark dataset validation

---

## Technology Stack

**AI / LLM**

- OpenAI API
- Multi-model inference
- LLM-as-a-Judge
- deterministic evaluation
- empirical model benchmarking

**Backend**

- Python
- FastAPI
- Pydantic
- HTTPX

**Data**

- SQLite
- versioned benchmark results

**UI / Observability**

- Streamlit
- Pandas

**Engineering**

- Pytest
- GitHub Actions
- environment-based configuration

---

## Engineering Principles

RouteGuard is built around several design principles:

**Evidence over assumptions**  
Do not assume an expensive model is automatically required.

**Quality before cost**  
A model must satisfy the quality constraint before cost optimization matters.

**Cost per success over cost per call**  
Retries, evaluation and fallbacks are part of inference economics.

**Uncertainty should be visible**  
Insufficient benchmark evidence results in explicit exploration mode.

**Evaluation should also be optimized**  
Do not spend more evaluating a trivial answer than generating it when objective validation is available.

**Self-evaluation is not truth verification**  
An LLM judge cannot independently establish factual truth without evidence.

**Routing should evolve with models**  
Empirical profiles can be regenerated as models, prices and workloads change.

---

## Current Status

RouteGuard is an engineering/portfolio project exploring **quality-aware and cost-aware LLM inference orchestration**.

Implemented:

- [x] Seven-model configurable registry
- [x] Task classification
- [x] Complexity analysis
- [x] Offline model benchmarking
- [x] Task-specific empirical model profiles
- [x] Wilson confidence bounds
- [x] Quality-constrained routing
- [x] Cost-per-success optimization
- [x] Explicit cold-start exploration
- [x] Deterministic evaluations
- [x] Independent LLM-as-a-Judge
- [x] Evaluation routing
- [x] Quality-triggered fallback
- [x] Token and cost accounting
- [x] SQLite telemetry
- [x] FastAPI service
- [x] Streamlit observability dashboard
- [x] Automated tests

In progress:

- [ ] Larger balanced benchmark corpus
- [ ] Held-out benchmark split
- [ ] Human calibration of LLM-judge scores
- [ ] Statistical comparison against routing baselines
- [ ] Measured cost/quality savings across all candidate models
- [ ] Learned success-probability routing

---

## Planned Experiments

The final evaluation will compare RouteGuard against several baselines:

```text
Always cheapest model
Always strongest model
Static task-based routing
Benchmark-driven RouteGuard
```

Primary metrics:

```text
Quality pass rate
Average cost per request
Cost per successful response
Average latency
Fallback rate
Token consumption
Quality-adjusted cost reduction
```

This will allow cost savings to be reported only after they are empirically measured rather than assumed.

---

## Future Direction

A natural extension is to replace aggregate task-level routing with a calibrated estimator:

```text
P(success | prompt, task, model)
```

The router could then optimize:

```text
argmin_model
    Expected Total Cost(model)

subject to
    P(success | prompt, model) >= Q_required
```

This would move RouteGuard from task-level empirical routing toward prompt-level adaptive model selection while preserving explainability and quality constraints.

---

## Project Goal

RouteGuard explores a practical question that becomes increasingly important as organizations deploy multiple foundation models:

> **How can an AI platform automatically choose the least expensive model that is sufficiently capable for each request—while providing measurable evidence that quality was preserved?**

The project focuses not just on model inference, but on the broader production problem of **routing, evaluation, uncertainty, observability, fallback, and inference economics**.

---

## Disclaimer

This project is an experimental engineering system, not a production SLA or guarantee of model quality.

Benchmark performance depends on dataset composition, model versions, provider behavior, pricing and evaluation methodology. LLM-as-a-Judge is itself fallible and should be calibrated against human-reviewed labels before production use.

No cost savings or quality improvements should be interpreted as established until validated on a sufficiently large held-out benchmark.