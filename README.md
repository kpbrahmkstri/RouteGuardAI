# RouteGuard AI — Seven-model GPT router

Selects from **seven configurable GPT model IDs** based on task type, complexity, context fit, estimated suitability, and **user-supplied pricing**. No manual small/medium/large labels. The independent judge, deterministic checks, fallbacks, FastAPI and Streamlit are inherited from RouteGuard v2.

## Run locally (no Docker)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env
uvicorn routeguard.api:app --reload
# second terminal, activate venv:
streamlit run dashboard.py
```

Visit http://localhost:8501 and http://localhost:8000/docs. Default `DEMO_MODE=true` simulates output and token usage; **no real inference or quality verification occurs in demo mode**. Set `DEMO_MODE=false`, configure `OPENAI_API_KEY` and confirm that your account has access to the selected models. For actual costs, populate `input_per_m` and `output_per_m` in `models.json` with current prices (USD per 1M tokens). Unconfigured prices are `null` and reported as unknown, never zero. If all prices are missing, the router picks the highest-suitability eligible candidate; it cannot claim cost optimality. Set `JUDGE_MODEL` and judge pricing in `.env`.

## Selection method

Task classification + prompt complexity -> filter context-incompatible models -> task-specific suitability prior -> threshold eligibility -> cheapest *priced* eligible candidate; when none qualifies, use highest-suitability model. Candidates with unknown prices are ranked after priced eligible candidates. On failed evaluation, try the next-ranked candidate, not a hardcoded larger tier. Seven model IDs are examples and may not all be available to your account. GPT-5 family support for Chat Completions and temperature parameters can vary by model/version; confirm endpoint compatibility for your chosen IDs before live use.

**Important:** initial `DEFAULT_PRIORS` are illustrative engineering heuristics, **not learned scores, calibrated probabilities, actual benchmarks or a guarantee of quality**. To earn cost/quality claims, benchmark all seven models on held-out tasks, populate `strengths` per model in `models.json` using measured results, and calibrate predictions. Cost figures exclude any usage not returned by the provider; judge usage is included when priced. Factual truthfulness is unverified without external evidence. In real mode, if no attempt passes, the last response is returned with status `unverified`.

## Customize models

Edit `models.json` to add/remove model IDs, prices and context capacities. Optional `strengths` object uses task keys `extraction`, `grounded_qa`, `coding`, `reasoning`, `summarization`, `general` with scores from 0 to 1. Current default entries omit prices deliberately; fill in verified prices before demonstrating dollar savings. The selection reason and all candidate estimates are visible in the dashboard and API response.

## Test

`pytest -q`

## Security / limitations

This is a portfolio MVP. No API keys are committed; do not submit sensitive prompts to unapproved providers. Context-based groundedness judge is fallible, and citation-format checks do not establish truth. GPT-5 model compatibility and pricing must be verified against your current account. No production SLA or calibrated prediction claims.

## Benchmark-driven routing (new)

**No invented benchmark results:** The dashboard starts without empirical scores. Populate `models.json` with current USD per-million-token input/output pricing, add an API key to `.env`, and set `DEMO_MODE=false`. Then run:

```bash
python -m routeguard.benchmark --dataset data/benchmark.jsonl
```

This performs **real calls across all seven configured models**, and optionally uses an independent judge. Expect API charges for all model and judge calls. You can limit expense with `--models gpt-4.1-mini gpt-5-mini`. To run deterministic-only benchmarking use `--no-judge`. A separate dataset with at least five representative examples **per task type** is needed before the conservative routing threshold can be used meaningfully. The included eight-case file is a pipeline smoke test, **not** a publishable quality benchmark. Build a versioned, held-out dataset with human-reviewed labels for resume-grade results.

The benchmark stores each attempt, objective checks, judge output, token usage, latency, and configured generation/judge costs in SQLite. `GET /benchmarks/profiles` and the Streamlit dashboard show empirical task-specific pass rates, sample counts, Wilson 95% lower bounds, average latency and average generation cost. A candidate only meets the router's quality constraint if it has at least `MIN_BENCHMARK_SAMPLES` cases and its lower bound clears the configured threshold. When no candidate qualifies, routing is explicitly marked **exploratory**, not quality-assured. Cold-start models use clearly labeled heuristic priors until benchmarked. The CLI reports failures without hiding them.

**Evaluation caveats:** The judge is fallible and must be calibrated against human labels. The included strict reference-match test may reject correct paraphrases; the sample dataset has intentionally simple references. A benchmark case with only required terms tests adherence, not factual correctness. Use held-out tests and human-reviewed ground truth to substantiate quality claims. Missing price fields mean cost optimization cannot be substantiated; benchmark averages exclude judge cost when ranking and actual execution reports include it. Do not claim production-readiness or proven savings until these are measured.

## Benchmark-first routing policy (v3)

RouteGuard no longer assigns hand-written capability/suitability scores to GPT models. It classifies each request (including `factual_qa`), retrieves task-specific empirical benchmark profiles, filters models using a conservative Wilson lower confidence bound against the configured quality threshold, and ranks qualifying models by total cost per successful answer. If there is insufficient benchmark evidence for the task, RouteGuard explicitly enters `exploration_cold_start` mode and uses known estimated price for calibration rather than pretending a model has measured quality. Run the benchmark suite before treating routing decisions as production-quality evidence.
