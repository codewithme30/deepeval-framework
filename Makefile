# DeepEval framework: handy shortcuts. Run `make help` to list them.
VENV    ?= .venv
PYTHON  ?= python3.11
BIN     := $(VENV)/bin
# `deepeval test run` = pytest + a DeepEval run report. Add -n 4 to run 4 workers in parallel.
EVAL    := $(BIN)/deepeval test run
PYTEST  := $(BIN)/pytest
WORKERS ?= 1

.PHONY: help install smoke rag agents conversational safety redteam benchmarks all report clean

help:
	@echo "install        create .venv and install pinned requirements"
	@echo "smoke          cheapest tests from every feature (what pull requests run)"
	@echo "rag            test_02: RAG metrics"
	@echo "agents         test_04: agent / tool-calling metrics"
	@echo "conversational test_06: multi-turn conversations"
	@echo "safety         test_08: bias, toxicity, PII, misuse, advice, role"
	@echo "redteam        test_09: DeepTeam red teaming (slow, spends tokens)"
	@echo "benchmarks     test_10: MMLU / TruthfulQA (live mode only, spends tokens)"
	@echo "all            everything except red team and benchmarks (the nightly suite)"
	@echo "report         run the nightly suite and write JUnit + DeepEval results into reports/"
	@echo "Tip: make all WORKERS=4   runs tests in parallel."

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env: add your OPENAI_API_KEY to it.")

smoke:
	$(EVAL) tests/ -m smoke -n $(WORKERS)

rag:
	$(EVAL) tests/test_02_rag_metrics.py -m rag -n $(WORKERS)

agents:
	$(EVAL) tests/test_04_agentic_metrics.py -m agent -n $(WORKERS)

conversational:
	$(EVAL) tests/test_06_conversational.py -m conversational -n $(WORKERS)

safety:
	$(EVAL) tests/test_08_safety_metrics.py -m safety -n $(WORKERS)

redteam:
	$(EVAL) tests/test_09_red_teaming.py -m redteam -n $(WORKERS)

# Benchmarks measure a real model, so they force live mode and opt in explicitly.
benchmarks:
	APP_MODE=live RUN_BENCHMARKS=1 $(EVAL) tests/test_10_benchmarks.py -m benchmark

all:
	$(EVAL) tests/ -m "not benchmark and not redteam" -n $(WORKERS)

report:
	@mkdir -p reports
	DEEPEVAL_RESULTS_FOLDER=reports/deepeval PYTEST_ADDOPTS="--junitxml=reports/junit.xml" \
		$(EVAL) tests/ -m "not benchmark and not redteam" -n $(WORKERS)
	@echo "Reports written to reports/ (junit.xml, deepeval/, redteam/)"

clean:
	rm -rf reports/* .pytest_cache .deepeval
	@touch reports/.gitkeep
