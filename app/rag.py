"""RAG pipeline: TF-IDF retriever over data/policies/ + an answer generator.

    question --> retrieve() --> top-k policy documents --> generate() --> answer

`query()` returns the answer AND the retrieved documents (`retrieval_context`), which is
exactly what DeepEval's RAG metrics need.

Stub mode: the retriever runs for real (it is deterministic); the generator looks the answer
up in app/stubs.py and falls back to an extractive sentence for unknown questions.
Live mode: the generator asks OpenAI to answer using only the retrieved documents.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from deepeval.tracing import observe, update_current_trace, update_retriever_span

from app import stubs
from app.tracing_hooks import attach

_STOPWORDS = frozenset(
    "a an the and or of to in on for is are be do does did my i you your we our it its at as by "
    "with from that this if how what when which who will can any no not than then so up out".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9$%]+", text.lower())
    # crude stemming: collapse plurals so "deductibles" matches "deductible"
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS]


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieval_context: list[str]
    sources: list[str] = field(default_factory=list)


class TfidfRetriever:
    """Tiny TF-IDF + cosine-similarity retriever (no vector DB needed)."""

    def __init__(self, policies_dir: str | Path):
        paths = sorted(Path(policies_dir).glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No policy documents found in {policies_dir}")
        self.names = [p.name for p in paths]
        self.docs = [p.read_text(encoding="utf-8").strip() for p in paths]
        doc_tokens = [Counter(_tokens(d)) for d in self.docs]
        n = len(self.docs)
        df = Counter(t for toks in doc_tokens for t in toks)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vectors = [self._vectorize(toks) for toks in doc_tokens]

    def _vectorize(self, counts: Counter) -> dict[str, float]:
        vec = {t: c * self._idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, question: str, top_k: int) -> list[tuple[str, str]]:
        """Return the top_k (filename, text) pairs, best first. Zero-score docs are dropped."""
        q = self._vectorize(Counter(_tokens(question)))
        scored = [(sum(w * dv.get(t, 0.0) for t, w in q.items()), i) for i, dv in enumerate(self._vectors)]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(self.names[i], self.docs[i]) for score, i in scored[:top_k] if score > 0]


def extractive_answer(question: str, documents: list[str]) -> str:
    """Fallback: return the retrieved sentence that shares the most words with the question."""
    q = set(_tokens(question))
    best, best_score = "", 0
    for doc in documents:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", doc):
            score = len(q & set(_tokens(sentence)))
            if sentence and not sentence.startswith("#") and score > best_score:
                best, best_score = sentence.strip(), score
    if not best:
        return "I'm sorry, I couldn't find that in our policy documents. Please contact a claims representative."
    return best


class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.top_k = config.get("app.rag_top_k", 3)
        self.mode = config.app_mode
        self.model = config.live_model
        self.retriever = TfidfRetriever(config.path("paths.policies_dir"))

    # ---- component 1: retriever -------------------------------------------------
    @observe(type="retriever")
    def retrieve(self, question: str) -> tuple[list[str], list[str]]:
        hits = self.retriever.search(question, self.top_k)
        documents = [text for _, text in hits]
        update_retriever_span(top_k=self.top_k)
        # For retriever-span metrics (e.g. contextual relevancy) the "output" is the documents.
        attach("retriever", input=question, actual_output="\n\n".join(documents), retrieval_context=documents)
        return documents, [name for name, _ in hits]

    # ---- component 2: generator -------------------------------------------------
    @observe(type="llm")
    def generate(self, question: str, documents: list[str]) -> str:
        if self.mode == "stub":
            answer = stubs.RAG_ANSWERS.get(stubs.normalize(question)) or extractive_answer(question, documents)
        else:
            from app.llm import chat

            context = "\n\n".join(documents)
            message = chat(
                [
                    {"role": "system", "content": "You are an insurance claims assistant. Answer ONLY using "
                     "the policy documents provided. If the answer is not in them, say you don't know."},
                    {"role": "user", "content": f"Policy documents:\n{context}\n\nQuestion: {question}"},
                ],
                model=self.model,
                temperature=0,
            )
            answer = message.content or ""
        attach("generator", input=question, actual_output=answer, retrieval_context=documents)
        return answer

    # ---- the full pipeline ------------------------------------------------------
    @observe()
    def query(self, question: str) -> RAGResult:
        documents, sources = self.retrieve(question)
        answer = self.generate(question, documents)
        # Trace-level view (used by end-to-end evals_iterator metrics): the user-visible input/output.
        update_current_trace(input=question, output=answer, retrieval_context=documents)
        return RAGResult(question=question, answer=answer, retrieval_context=documents, sources=sources)
