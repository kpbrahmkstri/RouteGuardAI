# RouteGuard AI

Quality-aware LLM routing MVP with explicit output checks, tier escalation, cost tracking, a FastAPI service, Streamlit dashboard, and a JSONL benchmark runner.

## Quick start

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn routeguard.api:app --reload
# In another terminal:
streamlit run dashboard.py
```

API: http://localhost:8000/docs; dashboard: http://localhost:8501.

```bash
curl -X POST http://localhost:8000/route -H 'Content-Type: application/json' -d '{"prompt":"Return JSON","expected_json":true}'
python benchmark.py
pytest -q
```

Docker: `cp .env.example .env && docker compose up --build`.

## Real model calls

Set `DEMO_MODE=false` and `OPENAI_API_KEY` in `.env`. Configure `OPENAI_BASE_URL` for a compatible provider, plus SMALL_MODEL, MEDIUM_MODEL, LARGE_MODEL and their input/output prices. Pricing defaults are **examples only** and must be verified before measuring savings. Some providers reject `temperature=0`; adjust `complete()` for their API. Do not commit `.env`.

The default medium and large model IDs are identical placeholders: change them to genuinely different models before benchmarking. Real mode sends prompts and context to the configured provider; never submit sensitive enterprise data without approval. The SQLite store logs answers and metadata; add authentication, redaction, encryption, rate limiting and durable storage before production use.

## What the checks mean

- `expected_answer`: normalized exact string match, not semantic correctness.
- `expected_json`: syntactically valid JSON, not schema adherence.
- `required_terms`: substring presence, not factual verification.
- `context`: verifies that citations like `[1]` reference existing paragraph indices, **not** factual grounding.
- Requests without any task-specific evaluation evidence return `unverified` with no answer. Failed checks escalate through configured tiers up to `max_attempts`; unresolved requests fail closed.
- `quality_threshold` currently changes starting tier at 0.95; it is **not** a calibrated success probability.

## Suggested next upgrades

1. Add JSON Schema and executable coding tests, and an evidence-entailment judge calibrated against human-reviewed examples.
2. Collect per-model results on disjoint train/validation/test splits; train a calibrated task-success predictor and optimize expected cost including retries and evaluator calls.
3. Compare fixed strongest, fixed cheapest, rule-based and learned routers on identical held-out cases. Record actual API bills, token usage, latency, first-pass success, final success and abstention rates.
4. Add provider-specific timeout/rate-limit handling, concurrency caps, circuit breakers and tracing.

## Project layout

- `routeguard/core.py`: heuristic router, model adapter, proxy checks, escalation and cost.
- `routeguard/api.py`: FastAPI endpoints and SQLite telemetry.
- `dashboard.py`: interactive Streamlit UI.
- `benchmark.py`: JSONL batch benchmark runner.
- `data/sample.jsonl`: tiny smoke-test dataset (not a representative benchmark).
- `tests/`: unit tests; `.github/workflows/ci.yml`: GitHub Actions.
