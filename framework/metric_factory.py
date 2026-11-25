"""Builds DeepEval metrics from config/thresholds.yaml. ONE place to tune.

    factory.build("faithfulness")                      # threshold + judge model come from config
    factory.build("faithfulness", strict_mode=True)    # extra DeepEval options pass straight through
    factory.g_eval("no_payout_promise", criteria=...)  # custom LLM-judged rule
    factory.build("misuse", domain="insurance claims") # metrics that need extra arguments

To add a metric: (1) add its class to REGISTRY below, (2) add `<key>: <threshold>` to
config/thresholds.yaml. No test code changes are needed to re-tune it afterwards.
"""
from __future__ import annotations

import inspect
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    BaseMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationalGEval,
    ConversationCompletenessMetric,
    DAGMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    JsonCorrectnessMetric,
    KnowledgeRetentionMetric,
    MisuseMetric,
    NonAdviceMetric,
    PatternMatchMetric,
    PIILeakageMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    RoleAdherenceMetric,
    RoleViolationMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ToxicityMetric,
    TurnRelevancyMetric,
)

from framework.config_loader import Config, Thresholds
from framework.model_wrappers import get_judge_model

# threshold key (in thresholds.yaml)  ->  DeepEval metric class
REGISTRY: dict[str, type] = {
    # RAG
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
    "hallucination": HallucinationMetric,
    # agents
    "task_completion": TaskCompletionMetric,
    "tool_correctness": ToolCorrectnessMetric,
    "argument_correctness": ArgumentCorrectnessMetric,
    "plan_quality": PlanQualityMetric,
    "plan_adherence": PlanAdherenceMetric,
    "step_efficiency": StepEfficiencyMetric,
    # conversations
    "turn_relevancy": TurnRelevancyMetric,
    "knowledge_retention": KnowledgeRetentionMetric,
    "role_adherence": RoleAdherenceMetric,
    "conversation_completeness": ConversationCompletenessMetric,
    # safety (score 1.0 = safe)
    "bias": BiasMetric,
    "toxicity": ToxicityMetric,
    "pii_leakage": PIILeakageMetric,
    "misuse": MisuseMetric,
    "non_advice": NonAdviceMetric,
    "role_violation": RoleViolationMetric,
    # deterministic (no LLM)
    "exact_match": ExactMatchMetric,
    "pattern_match": PatternMatchMetric,
}


class MetricFactory:
    def __init__(self, config: Config, thresholds: Thresholds):
        self.config = config
        self.thresholds = thresholds
        self._judge = None

    # ---- shared pieces ------------------------------------------------------------
    @property
    def judge(self):
        """The judge model, created on first use (so no OpenAI key is needed until then)."""
        if self._judge is None:
            self._judge = get_judge_model(self.config)
        return self._judge

    def threshold(self, key: str) -> float:
        return self.thresholds.get(key)

    def _defaults_for(self, cls: type, key: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        params = inspect.signature(cls.__init__).parameters
        kwargs.setdefault("threshold", self.threshold(key))
        if "model" in params and "model" not in kwargs:
            kwargs["model"] = self.judge
        if "async_mode" in params:
            kwargs.setdefault("async_mode", bool(self.config.get("evaluation.async_mode", True)))
        return kwargs

    # ---- builders -----------------------------------------------------------------
    def build(self, key: str, cls: type | None = None, **kwargs: Any) -> BaseMetric:
        """Build the metric registered under `key` (or `cls` if you pass your own class)."""
        cls = cls or REGISTRY.get(key)
        if cls is None:
            raise KeyError(f"'{key}' is not in metric_factory.REGISTRY. Known: {', '.join(sorted(REGISTRY))}")
        return cls(**self._defaults_for(cls, key, kwargs))

    def g_eval(self, key: str, name: str | None = None, **kwargs: Any) -> GEval:
        """G-Eval: an LLM judges the output against criteria OR explicit evaluation_steps."""
        return GEval(name=name or key, **self._defaults_for(GEval, key, kwargs))

    def conversational_g_eval(self, key: str, name: str | None = None, **kwargs: Any) -> ConversationalGEval:
        return ConversationalGEval(name=name or key, **self._defaults_for(ConversationalGEval, key, kwargs))

    def dag(self, key: str, dag: Any, name: str | None = None, **kwargs: Any) -> DAGMetric:
        return DAGMetric(name=name or key, dag=dag, **self._defaults_for(DAGMetric, key, kwargs))

    def json_correctness(self, schema: Any, key: str = "json_correctness", **kwargs: Any) -> JsonCorrectnessMetric:
        return JsonCorrectnessMetric(expected_schema=schema, **self._defaults_for(JsonCorrectnessMetric, key, kwargs))
