# deepeval-framework

A teachable, config-driven **DeepEval + pytest** framework that evaluates a mock *Insurance Claims
Assistant* (a RAG pipeline, a tool-calling agent and a multi-turn chatbot). It covers every core
DeepEval feature, one test file per feature, and runs in CI.

---

## 1. What is DeepEval? (for non-technical readers)

AI assistants answer in free text, so you cannot check them the way you check a calculator: there is
no single right string to compare against. **DeepEval** is a testing toolkit for exactly this. You
write a normal automated test that asks the assistant a question, then ask a second AI, called the
**judge**, to grade the answer on a quality such as "is it relevant?", "does it stick to our policy
documents?" or "is it polite?". The grade is a score between 0 and 1.

Each quality has a **pass mark** (a *threshold*). Score at or above it and the test passes; score
below it and the test fails, so a bad change is caught before customers see it. Because the checks are
ordinary tests, they run automatically on every code change, and the same tests can be re-run after
swapping to a different AI model to see whether quality went up or down.

## 2. Feature → file map

| # | File | Marker | What it covers |
|---|------|--------|----------------|
| 01 | `tests/test_01_pytest_basics.py` | `basics` | Test cases, `assert_test`, score / reason / `is_successful()`, several metrics at once, `evaluate()`, `strict_mode`, `include_reason` |
| 02 | `tests/test_02_rag_metrics.py` | `rag` | Answer relevancy, faithfulness, contextual precision / recall / relevancy, hallucination, RAGAS; *retriever problem* vs *generator problem* |
| 03 | `tests/test_03_custom_metrics.py` | `custom` | G-Eval (criteria and steps), DAG decision tree, custom `BaseMetric`, JSON schema, exact / pattern match |
| 04 | `tests/test_04_agentic_metrics.py` | `agent` | Tool and argument correctness, task completion, plan quality / adherence, step efficiency |
| 05 | `tests/test_05_tracing_component_evals.py` | `tracing` | `@observe` spans, metrics on individual spans, black-box vs component-level |
| 06 | `tests/test_06_conversational.py` | `conversational` | Turn relevancy, knowledge retention, role adherence, completeness, conversational G-Eval, `ConversationSimulator` |
| 07 | `tests/test_07_synthetic_datasets.py` | `dataset` | Goldens from JSON / CSV, `Synthesizer`, dataset evaluation, multi-turn goldens, Confident AI push / pull |
| 08 | `tests/test_08_safety_metrics.py` | `safety` | Bias, toxicity, PII leakage, misuse, non-advice, role violation |
| 09 | `tests/test_09_red_teaming.py` | `redteam`, `slow` | DeepTeam attacks (prompt injection, roleplay, ROT13) against bias / PII / prompt leakage / toxicity / excessive agency |
| 10 | `tests/test_10_benchmarks.py` | `benchmark`, `slow` | MMLU and TruthfulQA on a small, capped subset |

```
app/         the system under test: rag.py, agent.py, chatbot.py, tools.py, stubs.py (canned answers)
data/        policies/ (8 RAG source docs)  datasets/ (goldens.json, goldens.csv, conversations.json)
config/      settings.yaml (judge model, mode, cost caps)   thresholds.yaml (pass marks per metric)
framework/   config_loader, metric_factory, test_case_builders, model_wrappers, evals_runner, redteam
tests/       conftest.py + the ten feature files above
reports/     gitignored output (JUnit, DeepEval results, red-team results)
```

## 3. Setup in five steps (about 10 minutes)

1. **Python 3.11** is required. Clone the repo and `cd` into it.
2. `make install` (creates `.venv`, installs the pinned `requirements.txt`, copies `.env.example` to `.env`).
3. Put your key in `.env`: `OPENAI_API_KEY=sk-...`. Never commit `.env`.
4. `source .venv/bin/activate`
5. `make smoke` — the cheapest test from every file. All green means you are set up.

### Stub mode vs live mode

| | `stub` (default) | `live` |
|---|---|---|
| The app's answers | Canned, deterministic, from `app/stubs.py` (some are **deliberately bad**) | Real OpenAI calls (`live_model` in `settings.yaml`) |
| Needs `OPENAI_API_KEY` | **Yes, for judge-based tests.** The judge is always a real model | Yes |
| Use it for | Learning, CI, proving the metrics catch real defects | Measuring your real app and the benchmarks |

Switch with any of: `APP_MODE=live` in `.env`, `pytest --app-mode live`, or `app_mode:` in `config/settings.yaml`.
The designed-to-fail demos rely on the stub's bad answers, so in live mode they become *non-strict*
`xfail` (a good real answer will not break the suite).

Tests marked `no_judge` need no key at all (deterministic metrics, dataset loading, span structure).
Without a key everything else is **skipped with a message**, not crashed.

## 4. Run commands

| Feature | Marker command | Make target |
|---|---|---|
| Cheapest tests everywhere (PR gate) | `pytest -m smoke` | `make smoke` |
| 01 pytest basics | `pytest -m basics` | – |
| 02 RAG | `pytest -m rag` | `make rag` |
| 03 custom metrics | `pytest -m custom` | – |
| 04 agents | `pytest -m agent` | `make agents` |
| 05 tracing | `pytest -m tracing` | – |
| 06 conversations | `pytest -m conversational` | `make conversational` |
| 07 datasets | `pytest -m dataset` (`--max-goldens 2` to cap) | – |
| 08 safety | `pytest -m safety` | `make safety` |
| 09 red team | `pytest -m redteam` | `make redteam` |
| 10 benchmarks | `pytest -m benchmark --app-mode live --run-benchmarks` | `make benchmarks` |
| Nightly suite (no red team / benchmarks) | `pytest -m "not benchmark and not redteam"` | `make all` |
| Reports into `reports/` | – | `make report` |

**Parallel runs.** `deepeval test run` is pytest plus a DeepEval run report, and `-n` sets the number of
parallel workers: `deepeval test run tests/ -m smoke -n 4` (or `make all WORKERS=4`). Add `-m "<expr>"` to
filter by marker. The CLI cannot forward custom pytest flags, so use environment variables with it:
`APP_MODE=live RUN_BENCHMARKS=1 deepeval test run tests/test_10_benchmarks.py`.

## 5. How to read results

* **score** – 0 to 1 from the judge. **threshold** – the pass mark from `config/thresholds.yaml`.
  The test passes when `score >= threshold`. This holds for *every* metric, including safety ones
  (1.0 = safe).
* **reason** – the judge's one-sentence explanation. A failure message looks like
  `Metrics: Faithfulness (score: 0.25, threshold: 0.7, ..., reason: The answer claims flood damage is covered, but the documents say it is excluded) failed.`
* **`x` / xfail** – a test *designed to fail* to show a real failure (a hallucination, a wrong tool, a
  leaked SSN...). It is marked `xfail(strict=True, raises=AssertionError, ...)`, so the suite stays green.
  `raises=AssertionError` means only a genuine metric failure counts: a crash or an API error still
  shows up red instead of being hidden as "expected".
* **XPASS(strict)** – a designed-to-fail test *unexpectedly passed* (the judge was lenient or the app
  changed). That is reported as a failure on purpose, so you notice that a demo no longer demonstrates anything.
* **skipped** – usually a missing `OPENAI_API_KEY`, `CONFIDENT_API_KEY`, `--run-benchmarks`, or an optional package.

**Judge scores wobble.** An LLM grades the answers, so the same answer can score 0.82 one day and 0.78
the next. Assert with comfortable thresholds (0.5 to 0.7 here) rather than exact values, keep the judge
`temperature` at 0 (it is), and treat a metric hovering at the edge as a signal to look, not to nudge the
number. Set `evaluation.use_cache: true` to reuse identical results in `evaluate()` / `evals_iterator` runs (for `assert_test` use `deepeval test run -c`).

## 6. Extending the framework

**Add a test.** Put it in the file for its feature (never a new file per test). Build the test case with
a helper from `framework/test_case_builders.py`, get the metric from the factory, and assert:

```python
def test_answer_mentions_the_claim_deadline(rag_app, metric_factory):
    case = build_rag_test_case(rag_app, "How long do I have to report a claim?")
    assert_test(case, [metric_factory.build("answer_relevancy")])
```

To show a failure, add `@pytest.mark.xfail(strict=True, raises=AssertionError, reason="demonstrates ...")`
and, in stub mode, a deliberately bad answer in `app/stubs.py`.

**Add a metric.** (1) add `my_metric: 0.7` to `config/thresholds.yaml`; (2) register its class in
`REGISTRY` in `framework/metric_factory.py`; (3) use `metric_factory.build("my_metric", **extra_args)`.
For a custom `BaseMetric`, skip step 2 and call `metric_factory.build("my_metric", cls=MyMetric)`.
Re-tuning is then a one-line YAML change.

**Change the judge.** Edit `judge_model` in `config/settings.yaml` (or `JUDGE_MODEL=...`). No code changes.

## 7. Cost notes, version notes, dashboard

**Cost.** Every bulk activity is capped under `limits:` in `config/settings.yaml`: `max_goldens`,
`max_simulated_conversations`, `max_user_simulations`, `redteam_attacks_per_vulnerability`,
`redteam_max_vulnerability_types`, and the benchmark task / shot / problem counts. The expensive files
(`redteam`, `benchmark`, plus the synthesizer and simulator tests) carry the `slow` marker; run
`pytest -m "not slow"` to skip them. Judge calls use `gpt-4o-mini` by default. `evaluation.max_concurrent`
and `throttle_value` control rate-limit pressure. A smoke run is roughly a dozen small judge calls.

**Version notes** (pinned: `deepeval==4.3.5`, `deepteam==1.0.10`, `pytest==9.2.0`, `openai==3.59.0`).
The task description named some APIs from older releases; these are the current equivalents used here:

* `LLMTestCaseParams` / `TurnParams` → **`SingleTurnParams`** / **`MultiTurnParams`** (`from deepeval.test_case import ...`).
* **Safety metrics** (Bias, Toxicity, PIILeakage, Misuse, NonAdvice, RoleViolation) now score
  *1.0 = safe*, so a threshold is a **minimum** like every other metric (older releases used "lower is better").
* **RAGAS**: `RagasMetric` is no longer exported from `deepeval.metrics`; it lives in `deepeval.metrics.ragas`
  and needs `ragas`, `datasets` and `langchain-core` installed. That test skips itself otherwise.
* **`PatternMatchMetric`** uses `re.fullmatch`, so wrap a "contains" pattern as `(?s).*...*.*`.
* **Benchmarks**: `benchmark.overall_score` is set after `evaluate()` (the returned result exposes `overall_accuracy`).
  Models passed in must implement `generate(prompt, schema=None)`; see `framework/model_wrappers.py`.
* **Trace-level metrics** (`TaskCompletion`, `PlanQuality`, `PlanAdherence`, `StepEfficiency`) must run through tracing.
  DeepEval has two models that do not mix: plain `pytest` uses `dataset.evals_iterator(...)`; under
  `deepeval test run` a plugin wraps each test in its own trace and you use `assert_test(golden=..., metrics=...)`
  (and `evals_iterator` yields nothing there). `framework/evals_runner.py::run_trace_evals` picks the right path.
* **DeepTeam**: with `async_mode=True` the `model_callback` must be an `async def` (`framework/redteam.py::as_callback`
  builds the right kind); results are read from `risk_assessment.overview.vulnerability_type_results`.
  `deepteam 1.0.10` imports `sentry_sdk` without declaring it and breaks on `sentry-sdk>=3`, hence the pin.
* **Synthesizer**: `generate_goldens_from_docs` needs `chromadb`, `langchain-community`, `langchain-text-splitters`
  and `tiktoken` (pinned; `tiktoken` downloads its vocabulary from the internet on first use). Multi-turn
  generation needs a `ConversationalStylingConfig`. `EvaluationDataset.add_goldens_from_csv_file` needs `pandas`;
  benchmarks need `datasets` (and download from Hugging Face).
* **ConversationSimulator**'s callback takes `(input, turns)` and returns a `Turn`; the chatbot exposes a stateless
  `respond(message, history)` for it.
* `deepeval test run` accepts one path and `-m`, `-n`, `-c`, `-i` ...; it does not forward custom pytest options
  (use `APP_MODE`, `RUN_BENCHMARKS`).

**Confident AI dashboard (optional).** Create a free account, run `deepeval login` (or set
`CONFIDENT_API_KEY` in `.env`), then `deepeval test run ...` uploads each run for browsing and comparison.
With the key set, `test_07` also pushes/pulls the golden dataset (`deepeval-framework-demo-goldens`). In CI add
it as the `CONFIDENT_API_KEY` secret. Telemetry to DeepEval/DeepTeam is switched off in `conftest.py`.

**CI.** `.github/workflows/evals.yml`: pull requests run `deepeval test run tests/ -m smoke`; a nightly job
runs everything except benchmarks and red teaming (those two are weekly, so they are not duplicated nightly);
`reports/` is uploaded as a build artifact. Secrets: `OPENAI_API_KEY`, optional `CONFIDENT_API_KEY`.

## 8. Coming from Ragas?

**Overlap.** DeepEval's `AnswerRelevancy`, `Faithfulness`, `ContextualPrecision`, `ContextualRecall` and
`ContextualRelevancy` map onto Ragas' answer relevancy, faithfulness, context precision, context recall and
context relevance, and both score with an LLM judge. DeepEval can also call RAGAS itself (`test_02`).

**What DeepEval adds** (each is a file here):

| Capability | Where |
|---|---|
| Runs as plain pytest, `assert_test` fails the build, thresholds per metric | 01 |
| Custom rules: G-Eval, DAG decision trees, deterministic metrics | 03 |
| **Agents**: tool / argument correctness, task completion, plan and step quality | 04 |
| **Tracing** and span-level (component) evaluation | 05 |
| **Multi-turn conversations** and user simulation | 06 |
| Synthetic goldens and dataset management, dashboard sync | 07 |
| Safety metrics, plus **red teaming** (DeepTeam) | 08, 09 |
| Standard **benchmarks** (MMLU, TruthfulQA...) | 10 |

Rule of thumb: Ragas is a focused RAG scorer; DeepEval is a full test framework around scorers.
