"""Answer evaluation: exact match -> fuzzy match -> LLM-as-judge -> citations.

The ladder is deliberate and cheap-first:

1. **Exact match** on normalised text - catches identical strings.
2. **Fuzzy match** - containment either way on normalised text, which handles
   "Men's marathon" vs "Athletics at the 2008 Summer Olympics - Men's marathon".
3. **LLM-as-judge** (optional) - only invoked when deterministic matching fails
   and both a prediction and gold answer exist. Judges *semantic* equality
   ("N. Suleymanoglu" vs "Naim Suleymanoglu") without rewarding verbosity.
4. **Citation check** - independent of answer correctness: does the pipeline
   cite the gold documents?  Reported as precision/recall so the dashboard can
   separate "right answer, wrong evidence" from grounded correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from kg.textutil import normalize, similarity
from utils.metrics import TokenCounter

JUDGE_SYSTEM_PROMPT = (
    "You are a strict quiz grader. You are given a question, the gold answer "
    "and a predicted answer. Decide whether the prediction conveys the same "
    "fact as the gold answer. Ignore formatting, accents, transliteration "
    "spelling, and extra detail - but the core fact (name, number, title, "
    "date) must match. Never use outside knowledge to overrule the gold "
    "answer. Reply with JSON exactly like {\"same\": true, \"reason\": \"...\"}."
)


def judge_prompt(question: str, golds: List[str], pred: str) -> str:
    gold_block = "\n".join(f"- {g}" for g in golds)
    return (f"Question: {question}\n\nGold answer(s):\n{gold_block}\n\n"
            f"Predicted answer: {pred}\n\n"
            "Is the predicted answer the same fact as the gold answer? JSON:")


@dataclass
class Evaluation:
    """Outcome of grading one prediction against one gold answer set."""

    is_correct: bool = False
    match_type: str = "none"          # exact | fuzzy | llm | none
    accuracy_score: float = 0.0       # 0..1 (1 for exact/fuzzy/llm pass)
    similarity: float = 0.0           # best fuzzy similarity vs any gold
    citation_precision: Optional[float] = None   # cited ∩ gold / cited
    citation_recall: Optional[float] = None      # cited ∩ gold / gold
    citation_f1: Optional[float] = None          # 2 * P * R / (P + R)
    completeness: float = 0.0                    # token-level F1 / semantic completeness
    grounding_category: str = "unretrieved_failure" # grounded_correct | ungrounded_correct | grounded_incorrect | unretrieved_failure
    judge_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class Evaluator:
    """Grades pipeline predictions; optionally uses an LLM as a final judge."""

    def __init__(self, llm=None, use_llm_judge: bool = True,
                 judge_min_similarity: float = 0.25) -> None:
        self.llm = llm
        self.use_llm_judge = use_llm_judge and llm is not None \
            and getattr(llm, "available", False)
        self.judge_min_similarity = judge_min_similarity
        self.tokens = TokenCounter()
        self.judge_calls = 0
        self.judge_flips = 0     # deterministic-wrong -> llm-correct

    # ── deterministic layers ───────────────────────────────────────────
    @staticmethod
    def _best(golds: List[str], pred: str):
        """Return (match_type, best_similarity) for the deterministic ladder."""
        pn = normalize(pred)
        if not pn:
            return "none", 0.0
        best_sim = 0.0
        for g in golds:
            gn = normalize(g)
            if not gn:
                continue
            if pn == gn:
                return "exact", 1.0
            sim = similarity(pred, g)
            best_sim = max(best_sim, sim)
            # containment either way (same rule used in smoke tests)
            if len(gn) >= 2 and len(pn) >= 2 and (gn in pn or pn in gn):
                if gn.isdigit() or pn.isdigit():
                    if gn == pn:
                        return "exact", 1.0
                    continue
                return "fuzzy", max(sim, 0.95)
        return "none", best_sim

    @staticmethod
    def citation_overlap(citations: List[str], gold_doc_ids: List[str]):
        if not gold_doc_ids:
            return None, None
        gold = {d for d in gold_doc_ids if d}
        cited = {c for c in citations if c}
        if not cited:
            return 0.0, 0.0
        inter = len(cited & gold)
        return inter / len(cited), inter / len(gold)

    # ── main entry ─────────────────────────────────────────────────────
    def evaluate(self, question: str, golds: List[str], pred: str,
                 citations: List[str],
                 gold_doc_ids: Optional[List[str]] = None) -> Evaluation:
        ev = Evaluation()
        match, sim = self._best(golds or [], pred or "")
        ev.similarity = round(sim, 4)

        if match in ("exact", "fuzzy"):
            ev.is_correct = True
            ev.match_type = match
            ev.accuracy_score = 1.0
        elif self.use_llm_judge and (pred or "").strip() and golds \
                and sim >= self.judge_min_similarity:
            verdict = self._llm_judge(question, golds, pred)
            ev.judge_reason = verdict.get("reason", "")
            if verdict.get("same") is True:
                ev.is_correct = True
                ev.match_type = "llm"
                ev.accuracy_score = 1.0
                self.judge_flips += 1

        # Completeness (token-level semantic overlap / F1 vs gold answers)
        if golds and (pred or "").strip():
            pred_toks = set(normalize(pred).split())
            best_f1 = 0.0
            for g in golds:
                gold_toks = set(normalize(g).split())
                if pred_toks and gold_toks:
                    common = len(pred_toks & gold_toks)
                    f1 = (2.0 * common) / (len(pred_toks) + len(gold_toks))
                    best_f1 = max(best_f1, f1)
            ev.completeness = round(best_f1, 4)
        elif ev.is_correct:
            ev.completeness = 1.0

        p, r = self.citation_overlap(citations or [], gold_doc_ids or [])
        ev.citation_precision = None if p is None else round(p, 4)
        ev.citation_recall = None if r is None else round(r, 4)
        if p is not None and r is not None and (p + r) > 0:
            ev.citation_f1 = round((2.0 * p * r) / (p + r), 4)

        # Grounded correctness classification
        has_recall = (ev.citation_recall or 0.0) > 0.0
        if ev.is_correct and has_recall:
            ev.grounding_category = "grounded_correct"
        elif ev.is_correct and not has_recall:
            ev.grounding_category = "ungrounded_correct"
        elif not ev.is_correct and has_recall:
            ev.grounding_category = "grounded_incorrect"
        else:
            ev.grounding_category = "unretrieved_failure"
        return ev

    def _llm_judge(self, question: str, golds: List[str], pred: str) -> Dict[str, Any]:
        self.judge_calls += 1
        out = self.llm.complete_json(
            judge_prompt(question, golds, pred), JUDGE_SYSTEM_PROMPT,
            caller="eval.judge", counter=self.tokens, model=self.llm.eval_model,
            # Judging stays greedy even when the agent model explores, so the
            # score is reproducible across runs of a stochastic agent.
            temperature=getattr(self.llm, "eval_temperature", None))
        if not out:
            return {}
        same = out.get("same")
        if isinstance(same, str):
            same = same.strip().lower() in ("true", "yes", "1")
        return {"same": bool(same), "reason": str(out.get("reason", ""))[:300]}

    def stats(self) -> Dict[str, Any]:
        return {
            "llm_judge_enabled": self.use_llm_judge,
            "judge_calls": self.judge_calls,
            "judge_flips": self.judge_flips,
            "judge_tokens": self.tokens.total_tokens,
        }

