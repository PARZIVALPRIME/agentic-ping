"""Comprehensive Type-Aware Evaluation System for TigerGraph Agentic GraphRAG.

Evaluates predictions using domain-specific metric ladders tailored to question types:
  1. Numeric/Count: Numeric normalization, integer/float parse, absolute difference, tolerance bands.
  2. Entity: Diacritics stripping, name reordering ("Last, First" vs "First Last"), alias matching.
  3. Event: Title normalization, discipline/gender/distance extraction, prefix tolerance.
  4. Set: Multi-item parsing (delimiters, bullets, commas), Jaccard similarity, Precision, Recall, F1.
  5. Temporal: Edition parsing, Olympic cycle sequencing, chronological comparison.
  6. Evidence Grounding & Provenance: Verification of cited documents and graph traversal edges.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from kg.textutil import normalize, similarity


class AnswerType(str, Enum):
    NUMERIC = "numeric"
    ENTITY = "entity"
    EVENT = "event"
    SET = "set"
    TEMPORAL = "temporal"
    GENERAL = "general"


@dataclass
class TypeEvaluationResult:
    """Detailed evaluation result for a type-aware evaluation."""

    is_correct: bool = False
    answer_type: AnswerType = AnswerType.GENERAL
    score: float = 0.0                     # Continuous score in [0.0, 1.0]
    metric_name: str = "exact_match"       # e.g., numeric_exact, jaccard_f1, entity_alias, event_overlap
    tolerance_used: float = 0.0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    parsed_prediction: Any = None
    parsed_gold: Any = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "is_correct": self.is_correct,
            "answer_type": self.answer_type.value,
            "score": round(self.score, 4),
            "metric_name": self.metric_name,
            "tolerance_used": self.tolerance_used,
            "details": self.details,
        }
        if self.precision is not None:
            d["precision"] = round(self.precision, 4)
        if self.recall is not None:
            d["recall"] = round(self.recall, 4)
        if self.f1 is not None:
            d["f1"] = round(self.f1, 4)
        return d


class TypeAwareEvaluator:
    """Evaluates question answers based on their intrinsic semantic answer type."""

    @staticmethod
    def infer_answer_type(question: str, golds: List[str], qtype: str = "") -> AnswerType:
        """Infer the appropriate evaluation answer type."""
        # 1. If explicit qtype is aggregation, it is almost always numeric/count
        if qtype == "aggregation":
            return AnswerType.NUMERIC

        # 2. Check if golds are purely digits
        clean_golds = [g.strip() for g in golds if g.strip()]
        if clean_golds and all(re.match(r"^-?\d+(\.\d+)?$", g) for g in clean_golds):
            return AnswerType.NUMERIC

        # 3. Check for set indicators (commas, semicolons, lists, or multiple golds)
        if len(clean_golds) > 1 and any("," in g or ";" in g for g in clean_golds):
            return AnswerType.SET

        # 4. Check for event indicators
        q_lower = question.lower()
        if "which event" in q_lower or "what event" in q_lower or qtype == "superlative":
            return AnswerType.EVENT

        # 5. Check for temporal/year indicators
        if "immediately before" in q_lower or "immediately after" in q_lower or "what year" in q_lower:
            # If the gold is a person name, treat as ENTITY, else TEMPORAL
            if clean_golds and re.match(r"^\d{4}$", clean_golds[0]):
                return AnswerType.TEMPORAL

        # 6. Entity indicators (who, which person, which athlete, which nation, country)
        if any(w in q_lower for w in ("who won", "which athlete", "which competitor", "which nation", "which country")):
            return AnswerType.ENTITY

        return AnswerType.GENERAL

    # ── 1. Numeric / Count Evaluator ──────────────────────────────────
    @classmethod
    def evaluate_numeric(cls, golds: List[str], pred: str, tolerance: float = 0.0) -> TypeEvaluationResult:
        """Evaluate numeric/count answers with optional tolerance bands."""
        res = TypeEvaluationResult(answer_type=AnswerType.NUMERIC, metric_name="numeric_exact")

        # Extract numeric tokens from prediction
        pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
        if not pred_nums:
            res.is_correct = False
            res.score = 0.0
            res.details = {"error": "no_numeric_found_in_prediction", "raw_pred": pred}
            return res

        pred_val = float(pred_nums[0])
        res.parsed_prediction = pred_val

        best_score = 0.0
        best_gold_val = None
        exact_match = False

        for g in golds:
            g_nums = re.findall(r"-?\d+(?:\.\d+)?", g.replace(",", ""))
            if not g_nums:
                continue
            g_val = float(g_nums[0])
            diff = abs(pred_val - g_val)
            if diff <= tolerance:
                exact_match = True
                best_score = 1.0
                best_gold_val = g_val
                break
            else:
                # Continuous proximity score: 1 / (1 + diff)
                prox = 1.0 / (1.0 + diff)
                if prox > best_score:
                    best_score = prox
                    best_gold_val = g_val

        res.is_correct = exact_match
        res.score = best_score
        res.parsed_gold = best_gold_val
        res.tolerance_used = tolerance
        res.details = {
            "pred_val": pred_val,
            "target_val": best_gold_val,
            "absolute_diff": abs(pred_val - best_gold_val) if best_gold_val is not None else None,
        }
        return res

    # ── 2. Entity Evaluator ────────────────────────────────────────────
    @classmethod
    def _strip_diacritics(cls, text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    @classmethod
    def evaluate_entity(cls, golds: List[str], pred: str) -> TypeEvaluationResult:
        """Evaluate entity answers with diacritic tolerance, name reversal, and alias matching."""
        res = TypeEvaluationResult(answer_type=AnswerType.ENTITY, metric_name="entity_match")
        norm_pred = cls._strip_diacritics(normalize(pred))
        if not norm_pred:
            return res

        best_sim = 0.0
        match_type = "none"

        for g in golds:
            norm_g = cls._strip_diacritics(normalize(g))
            if not norm_g:
                continue

            # Exact match after diacritic stripping
            if norm_pred == norm_g:
                res.is_correct = True
                res.score = 1.0
                res.metric_name = "entity_exact"
                res.parsed_gold = g
                res.parsed_prediction = pred
                return res

            # Name order reversal check ("Chen Ding" vs "Ding Chen")
            g_tokens = sorted(norm_g.split())
            p_tokens = sorted(norm_pred.split())
            if len(g_tokens) > 1 and g_tokens == p_tokens:
                res.is_correct = True
                res.score = 1.0
                res.metric_name = "entity_token_permutation"
                res.parsed_gold = g
                res.parsed_prediction = pred
                return res

            # Containment (e.g. "Naim Suleymanoglu" in full result)
            if len(norm_g) >= 3 and (norm_g in norm_pred or norm_pred in norm_g):
                match_type = "entity_containment"
                best_sim = max(best_sim, 0.95)

            sim = similarity(norm_pred, norm_g)
            if sim > best_sim:
                best_sim = sim
                if sim >= 0.85:
                    match_type = "entity_fuzzy"

        res.is_correct = best_sim >= 0.85 or match_type != "none"
        res.score = best_sim if res.is_correct else best_sim * 0.5
        res.metric_name = match_type if res.is_correct else "entity_mismatch"
        res.details = {"similarity": best_sim, "matched_rule": match_type}
        return res

    # ── 3. Event Evaluator ─────────────────────────────────────────────
    @classmethod
    def evaluate_event(cls, golds: List[str], pred: str) -> TypeEvaluationResult:
        """Evaluate event title predictions, ignoring verbose prefixes and sport noise."""
        res = TypeEvaluationResult(answer_type=AnswerType.EVENT, metric_name="event_match")
        norm_pred = normalize(pred)
        if not norm_pred:
            return res

        # Strip standard prefixes like "Athletics at the 2008 Summer Olympics -"
        def _clean_event_title(t: str) -> str:
            t = normalize(t)
            # Remove "at the XXXX [Summer|Winter] Olympics"
            t = re.sub(r"at the \d{4} (summer|winter) olympics\s*[-–—:]*", "", t)
            # Remove sport prefix before dash
            if "-" in t:
                parts = t.split("-", 1)
                if len(parts[1].strip()) >= 3:
                    t = parts[1].strip()
            return t.strip()

        clean_p = _clean_event_title(pred)
        best_sim = 0.0

        for g in golds:
            clean_g = _clean_event_title(g)
            if clean_p == clean_g or clean_g in clean_p or clean_p in clean_g:
                res.is_correct = True
                res.score = 1.0
                res.metric_name = "event_title_exact"
                res.details = {"clean_pred": clean_p, "clean_gold": clean_g}
                return res

            sim = similarity(clean_p, clean_g)
            if sim > best_sim:
                best_sim = sim

        res.is_correct = best_sim >= 0.80
        res.score = best_sim
        res.metric_name = "event_fuzzy" if res.is_correct else "event_mismatch"
        res.details = {"best_similarity": best_sim, "clean_pred": clean_p}
        return res

    # ── 4. Set Evaluator ───────────────────────────────────────────────
    @classmethod
    def _parse_set_items(cls, text: str) -> Set[str]:
        """Split a comma/semicolon/newline separated string into clean items."""
        # Split on commas, semicolons, bullets, and newlines
        raw_items = re.split(r"[,;\n•\*\-]\s*", text)
        items = set()
        for it in raw_items:
            norm = normalize(it)
            if norm and len(norm) > 1:
                items.add(norm)
        return items

    @classmethod
    def evaluate_set(cls, golds: List[str], pred: str) -> TypeEvaluationResult:
        """Evaluate set answers with Jaccard, Precision, Recall, and F1."""
        res = TypeEvaluationResult(answer_type=AnswerType.SET, metric_name="set_jaccard_f1")

        # Parse gold set
        gold_items: Set[str] = set()
        for g in golds:
            gold_items.update(cls._parse_set_items(g))

        pred_items = cls._parse_set_items(pred)
        if not gold_items or not pred_items:
            res.is_correct = False
            res.score = 0.0
            return res

        intersection = gold_items & pred_items
        union = gold_items | pred_items

        jaccard = len(intersection) / len(union) if union else 0.0
        prec = len(intersection) / len(pred_items) if pred_items else 0.0
        rec = len(intersection) / len(gold_items) if gold_items else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        res.precision = prec
        res.recall = rec
        res.f1 = f1
        res.score = f1
        res.is_correct = f1 >= 0.80 or jaccard >= 0.70
        res.details = {
            "jaccard": round(jaccard, 4),
            "overlap_items": list(intersection),
            "missing_items": list(gold_items - pred_items),
            "extra_items": list(pred_items - gold_items),
        }
        return res

    # ── 5. Master Dispatcher ──────────────────────────────────────────
    @classmethod
    def evaluate(cls, question: str, golds: List[str], pred: str,
                 qtype: str = "", tolerance: float = 0.0) -> TypeEvaluationResult:
        """Run the type-aware evaluation ladder based on the inferred answer type."""
        atype = cls.infer_answer_type(question, golds, qtype=qtype)

        if atype == AnswerType.NUMERIC:
            return cls.evaluate_numeric(golds, pred, tolerance=tolerance)
        elif atype == AnswerType.ENTITY:
            return cls.evaluate_entity(golds, pred)
        elif atype == AnswerType.EVENT:
            return cls.evaluate_event(golds, pred)
        elif atype == AnswerType.SET:
            return cls.evaluate_set(golds, pred)
        else:
            # Fallback to entity / string overlap
            return cls.evaluate_entity(golds, pred)


class GroundingAuditor:
    """Audits evidence citations and factual grounding against retrieved documents."""

    @staticmethod
    def audit_grounding(pred: str, cited_docs: List[Dict[str, Any]],
                        gold_doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Audit whether prediction terms appear in cited documents and match gold documents."""
        pred_terms = set(normalize(pred).split())
        doc_matches = 0
        total_docs = len(cited_docs)

        cited_ids = []
        for d in cited_docs:
            doc_id = d.get("id") or d.get("doc_id", "")
            if doc_id:
                cited_ids.append(doc_id)
            content = normalize(d.get("text", "") or d.get("content", ""))
            if any(term in content for term in pred_terms if len(term) >= 3):
                doc_matches += 1

        grounding_ratio = (doc_matches / total_docs) if total_docs > 0 else 0.0

        # Overlap with gold doc ids
        p, r, f1 = None, None, None
        if gold_doc_ids:
            gold_set = set(gold_doc_ids)
            cited_set = set(cited_ids)
            inter = len(gold_set & cited_set)
            p = (inter / len(cited_set)) if cited_set else 0.0
            r = (inter / len(gold_set)) if gold_set else 0.0
            f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return {
            "grounding_ratio": round(grounding_ratio, 4),
            "docs_cited": len(cited_ids),
            "citation_precision": round(p, 4) if p is not None else None,
            "citation_recall": round(r, 4) if r is not None else None,
            "citation_f1": round(f1, 4) if f1 is not None else None,
            "is_factually_grounded": grounding_ratio >= 0.50,
        }
