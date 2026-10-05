"""Versioned offline model benchmarks.

Only real, evaluated model calls populate benchmark profiles.

Benchmark evaluation is task-aware:
- Objective tasks can use deterministic evaluation without an LLM judge.
- Subjective / semantic tasks can use an independent LLM-as-a-Judge.
- Generation and evaluation costs are tracked separately.
- Model profiles are maintained per (model, task).
"""

import asyncio
import json
import math
import os
import sqlite3
import time
from pathlib import Path

from .core import (
    models,
    analyze,
    generate,
    deterministic,
    judge,
    price,
    needs_llm_judge,
)


DATASET_VERSION = "v3"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def connection():
    """Create/open the RouteGuard benchmark database."""

    db = sqlite3.connect(
        os.getenv("ROUTEGUARD_DB", "routeguard.db"),
        timeout=30,
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            case_id TEXT NOT NULL,
            model TEXT NOT NULL,
            task TEXT NOT NULL,
            passed INTEGER NOT NULL,
            generation_cost REAL,
            judge_cost REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_s REAL,
            details TEXT,
            created REAL NOT NULL
        )
        """
    )

    db.commit()
    return db


# ---------------------------------------------------------------------------
# Statistical confidence
# ---------------------------------------------------------------------------

def wilson_lower_bound(successes, n, z=1.96):
    """
    Conservative lower confidence bound for a binomial success rate.

    RouteGuard uses this instead of raw pass rate when determining whether
    benchmark evidence is strong enough to satisfy a quality threshold.
    """

    if n <= 0:
        return 0.0

    p = successes / n
    denominator = 1 + (z * z / n)

    numerator = (
        p
        + z * z / (2 * n)
        - z
        * math.sqrt(
            (p * (1 - p) + z * z / (4 * n)) / n
        )
    )

    return max(0.0, numerator / denominator)


# ---------------------------------------------------------------------------
# Benchmark profiles
# ---------------------------------------------------------------------------

def load_profiles():
    """
    Load empirical benchmark profiles grouped by model and task.

    Profiles contain:
    - sample count
    - successful responses
    - mean generation cost
    - mean total cost
    - mean latency

    Total cost includes both generation and LLM evaluation when an
    LLM judge was required.
    """

    db = connection()

    try:
        rows = db.execute(
            """
            SELECT
                model,
                task,
                COUNT(*),
                SUM(passed),
                AVG(generation_cost),
                AVG(
                    COALESCE(generation_cost, 0)
                    + COALESCE(judge_cost, 0)
                ),
                AVG(latency_s)
            FROM benchmark_results
            WHERE dataset_version = ?
            GROUP BY model, task
            """,
            (DATASET_VERSION,),
        ).fetchall()

    finally:
        db.close()

    return {
        (model, task): {
            "n": n,
            "passes": passes,
            "mean_generation_cost_usd": generation_cost,
            "mean_total_cost_usd": total_cost,
            "mean_latency_s": latency,
        }
        for (
            model,
            task,
            n,
            passes,
            generation_cost,
            total_cost,
            latency,
        ) in rows
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def dataset(path):
    """
    Load and validate the benchmark JSONL dataset.

    Every benchmark case must have:
    - a unique ID
    - a prompt
    - at least one independently checkable evaluation criterion

    Checkable criteria include:
    - expected_answer
    - required_terms
    - expected_json
    - context
    """

    cases = []
    ids = set()

    for line in Path(path).read_text(encoding="utf8").splitlines():

        if not line.strip():
            continue

        case = json.loads(line)
        case_id = case["id"]

        if case_id in ids:
            raise ValueError(
                f"Duplicate benchmark case {case_id}"
            )

        ids.add(case_id)

        if not case.get("prompt"):
            raise ValueError(
                f"Benchmark case has no prompt: {case_id}"
            )

        has_checkable_criterion = any(
            [
                case.get("expected_answer") is not None,
                bool(case.get("required_terms")),
                bool(case.get("expected_json")),
                bool(case.get("context")),
            ]
        )

        if not has_checkable_criterion:
            raise ValueError(
                "Benchmark requires independently checkable criteria: "
                + case_id
            )

        cases.append(case)

    if not cases:
        raise ValueError("Empty benchmark dataset")

    return cases


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

async def run_benchmark(
    path="data/benchmark.jsonl",
    model_ids=None,
    run_id=None,
    threshold=0.85,
    use_judge=True,
):
    """
    Benchmark configured models against the supplied dataset.

    Evaluation strategy is selected per task.

    Example:

    factual_qa + expected_answer
        -> deterministic evaluation

    structured_output + expected_json
        -> deterministic evaluation

    knowledge_explanation
        -> deterministic checks + LLM judge

    reasoning / summarization / grounded QA
        -> deterministic checks + LLM judge

    The benchmark stores results per:
        model × task × benchmark case
    """

    if os.getenv("DEMO_MODE", "true").lower() == "true":
        raise ValueError(
            "Benchmarks require DEMO_MODE=false "
            "and actual provider calls"
        )

    cases = dataset(path)

    available = {
        model.model: model
        for model in models()
    }

    selected = model_ids or list(available)

    unknown_models = set(selected) - set(available)

    if unknown_models:
        raise ValueError(
            "Unknown models: " + str(unknown_models)
        )

    run_id = run_id or str(int(time.time()))

    db = connection()
    summary = []

    try:

        for case in cases:

            # ---------------------------------------------------------------
            # Determine task
            # ---------------------------------------------------------------

            task = case.get("task")

            if not task:
                task = analyze(
                    case["prompt"],
                    case.get("context", ""),
                    case.get("expected_json", False),
                )["task"]

            # ---------------------------------------------------------------
            # Run same benchmark case against each selected model
            # ---------------------------------------------------------------

            for model_id in selected:

                model = available[model_id]
                start = time.monotonic()

                try:

                    # -------------------------------------------------------
                    # 1. Generate candidate response
                    # -------------------------------------------------------

                    answer, inp, out, generation_latency = await generate(
                        model,
                        case["prompt"],
                        case.get("context", ""),
                        case.get("expected_json", False),
                    )

                    # -------------------------------------------------------
                    # 2. Deterministic evaluation
                    # -------------------------------------------------------

                    checks = deterministic(
                        answer,
                        case.get("context", ""),
                        case.get("expected_answer"),
                        case.get("expected_json", False),
                        case.get("required_terms"),
                    )

                    mandatory_pass = all(
                        check["passed"]
                        for check in checks.values()
                    )

                    # -------------------------------------------------------
                    # 3. Decide whether an LLM judge is necessary
                    # -------------------------------------------------------

                    judge_required = (
                        use_judge
                        and needs_llm_judge(
                            task,
                            expected_answer=case.get(
                                "expected_answer"
                            ),
                            expected_json=case.get(
                                "expected_json",
                                False,
                            ),
                            required_terms=case.get(
                                "required_terms"
                            ),
                            context=case.get(
                                "context",
                                "",
                            ),
                        )
                    )

                    # -------------------------------------------------------
                    # 4. Semantic evaluation if required
                    # -------------------------------------------------------

                    if judge_required:

                        evaluation = await judge(
                            case["prompt"],
                            answer,
                            case.get("context", ""),
                            case.get("expected_answer"),
                        )

                        scores = {
                            key: value
                            for key, value
                            in evaluation.get(
                                "scores",
                                {},
                            ).items()
                            if key != "truthfulness"
                            and value is not None
                        }

                        evaluation_pass = (
                            evaluation.get("status")
                            == "evaluated"
                            and bool(scores)
                            and all(
                                score >= threshold
                                for score in scores.values()
                            )
                        )

                        evaluation_strategy = "llm_judge"

                    else:

                        # Objective criteria already determine success.
                        evaluation = {
                            "status": "not_required",
                            "scores": {},
                            "explanations": {
                                "reason":
                                    "Deterministic evaluation "
                                    "is sufficient for this "
                                    "benchmark case."
                            },
                            "evidence":
                                "Independent objective validation "
                                "criteria were supplied.",
                            "cost_usd": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "latency_s": 0,
                        }

                        evaluation_pass = True
                        evaluation_strategy = "deterministic"

                    # -------------------------------------------------------
                    # 5. Final benchmark decision
                    # -------------------------------------------------------

                    passed = (
                        mandatory_pass
                        and evaluation_pass
                    )

                    # -------------------------------------------------------
                    # 6. Cost accounting
                    # -------------------------------------------------------

                    generation_cost = price(
                        model,
                        inp,
                        out,
                    )

                    judge_cost = (
                        evaluation.get("cost_usd", 0)
                        or 0
                    )

                    total_cost = (
                        generation_cost + judge_cost
                        if generation_cost is not None
                        else None
                    )

                    total_input_tokens = (
                        inp
                        + evaluation.get(
                            "input_tokens",
                            0,
                        )
                    )

                    total_output_tokens = (
                        out
                        + evaluation.get(
                            "output_tokens",
                            0,
                        )
                    )

                    # End-to-end latency includes generation + evaluation.
                    total_latency = (
                        time.monotonic() - start
                    )

                    # -------------------------------------------------------
                    # 7. Store rich benchmark details
                    # -------------------------------------------------------

                    details = {
                        "answer": answer,
                        "task": task,
                        "evaluation_strategy":
                            evaluation_strategy,
                        "judge_required":
                            judge_required,
                        "checks": checks,
                        "judge": evaluation,
                        "generation": {
                            "input_tokens": inp,
                            "output_tokens": out,
                            "cost_usd":
                                generation_cost,
                            "latency_s":
                                generation_latency,
                        },
                        "total": {
                            "input_tokens":
                                total_input_tokens,
                            "output_tokens":
                                total_output_tokens,
                            "tokens":
                                total_input_tokens
                                + total_output_tokens,
                            "cost_usd":
                                total_cost,
                            "latency_s":
                                total_latency,
                        },
                    }

                except Exception as exc:

                    passed = False
                    generation_cost = None
                    judge_cost = None
                    inp = 0
                    out = 0

                    total_latency = (
                        time.monotonic() - start
                    )

                    evaluation_strategy = "error"

                    details = {
                        "error": str(exc)[:300],
                        "task": task,
                    }

                # -----------------------------------------------------------
                # 8. Persist benchmark result
                # -----------------------------------------------------------

                db.execute(
                    """
                    INSERT INTO benchmark_results(
                        run_id,
                        dataset_version,
                        case_id,
                        model,
                        task,
                        passed,
                        generation_cost,
                        judge_cost,
                        input_tokens,
                        output_tokens,
                        latency_s,
                        details,
                        created
                    )
                    VALUES(
                        ?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    (
                        run_id,
                        DATASET_VERSION,
                        case["id"],
                        model_id,
                        task,
                        int(passed),
                        generation_cost,
                        judge_cost,
                        inp,
                        out,
                        total_latency,
                        json.dumps(details),
                        time.time(),
                    ),
                )

                db.commit()

                summary.append(
                    {
                        "case_id": case["id"],
                        "model": model_id,
                        "task": task,
                        "passed": passed,
                        "evaluation_strategy":
                            evaluation_strategy,
                        "generation_cost_usd":
                            generation_cost,
                        "judge_cost_usd":
                            judge_cost,
                        "total_cost_usd":
                            (
                                generation_cost
                                + (judge_cost or 0)
                                if generation_cost
                                is not None
                                else None
                            ),
                        "latency_s":
                            round(
                                total_latency,
                                3,
                            ),
                        "error":
                            details.get("error"),
                    }
                )

    finally:
        db.close()

    return {
        "run_id": run_id,
        "dataset_version": DATASET_VERSION,
        "results": summary,
        "profiles": profile_rows(),
    }


# ---------------------------------------------------------------------------
# Profile reporting
# ---------------------------------------------------------------------------

def profile_rows():
    """
    Return benchmark profiles in dashboard/API-friendly format.
    """

    rows = []

    for (model, task), values in sorted(
        load_profiles().items()
    ):

        n = values["n"]
        passes = values["passes"]

        pass_rate = (
            passes / n
            if n
            else 0
        )

        lower_bound = wilson_lower_bound(
            passes,
            n,
        )

        mean_total_cost = values[
            "mean_total_cost_usd"
        ]

        cost_per_success = None

        if (
            mean_total_cost is not None
            and pass_rate > 0
        ):
            cost_per_success = (
                mean_total_cost
                / pass_rate
            )

        rows.append(
            {
                "model": model,
                "task": task,
                "samples": n,
                "passes": passes,
                "pass_rate":
                    round(pass_rate, 3),
                "wilson_lower_bound":
                    round(lower_bound, 3),
                "mean_generation_cost_usd":
                    values[
                        "mean_generation_cost_usd"
                    ],
                "mean_total_cost_usd":
                    mean_total_cost,
                "cost_per_success_usd":
                    cost_per_success,
                "mean_latency_s":
                    values[
                        "mean_latency_s"
                    ],
            }
        )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Run RouteGuard offline model benchmarks."
        )
    )

    parser.add_argument(
        "--dataset",
        default="data/benchmark.jsonl",
    )

    parser.add_argument(
        "--models",
        nargs="*",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "Disable LLM-as-a-Judge and rely only "
            "on deterministic benchmark criteria."
        ),
    )

    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            path=args.dataset,
            model_ids=args.models,
            threshold=args.threshold,
            use_judge=not args.no_judge,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )