"""
Scope qualifier detection for Lift Mind queries.

When a technician asks a question with a spatial / component scope qualifier
("in the shaft", "on the controller", "at the SEC J board", "in the cabin"),
the retrieval system should prefer chunks that match that scope.

This module:
  1. Detects scope qualifiers in the user query.
  2. Returns a list of additional BM25 keywords to inject.
  3. Returns a list of "preferred" terms that the reranker can use for boosts.

It is intentionally simple — no ML, no LLM call. The patterns are tuned for
elevator/lift troubleshooting language.

USAGE
-----

    from liftmind.scope_qualifier import detect_scope, scope_keywords

    scope = detect_scope("What components in the shaft should I inspect for NRUN?")
    # -> {'scope': 'shaft', 'keywords': ['shaft', 'sensor', 'magnet', 'guide rail', ...],
    #     'penalise': ['machine room', 'controller cabinet', ...]}

    if scope:
        keyword_queries.extend(scope['keywords'])
"""
from __future__ import annotations
import re
from typing import Optional, Dict, List


# Each scope: trigger phrases (regex), boost terms (added to BM25), penalise terms
# (other-area words that should DEMOTE a chunk if they appear without the scope term).
SCOPE_DEFINITIONS: List[Dict] = [
    {
        "scope": "shaft",
        "triggers": [
            r"\bin\s+the\s+shaft\b",
            r"\bshaft[-\s]?side\b",
            r"\bshaft\s+(?:component|sensor|wiring|magnet)",
            r"\bin\s+the\s+well\b",  # UK / older terminology
            r"\bhoistway\b",
        ],
        "keywords": [
            "shaft", "hoistway", "magnet", "bistable", "FAI", "FAS", "ZP",
            "AGB", "AGH", "limit switch", "guide rail", "inductive sensor",
            "speed exchange", "slow-down magnet", "leveling magnet",
        ],
        "penalise": [
            "controller cabinet", "machine room", "MRL panel", "main board",
        ],
    },
    {
        "scope": "controller",
        "triggers": [
            r"\bat\s+the\s+controller\b",
            r"\bon\s+the\s+(?:main\s+)?(?:PCB|board|controller)\b",
            r"\bMINIPAD\b",
            r"\bSEC\s*J\s+board\b",
            r"\bWE1200\b",
            r"\bEasyMax\b",
            r"\bDMG\s+Junior\b",
        ],
        "keywords": [
            "controller", "PCB", "main board", "MINIPAD", "SEC J", "terminal",
            "parameter", "FUN1", "FUN2", "FUN3", "FUN4", "FUN5", "FUN6",
            "DIP switch", "jumper", "LED indicator", "relay output",
        ],
        "penalise": [
            "shaft", "hoistway", "guide rail",
        ],
    },
    {
        "scope": "machine_room",
        "triggers": [
            r"\bin\s+the\s+machine\s+room\b",
            r"\bmachine[-\s]?room[-\s]?less\b",  # MRL
            r"\bat\s+the\s+pump\s+unit\b",
            r"\bon\s+the\s+motor\b",
        ],
        "keywords": [
            "machine room", "pump", "motor", "valve", "hydraulic unit",
            "rupture valve", "GMV", "VC 3006", "tank", "oil",
            "main contactor", "contactor",
        ],
        "penalise": [
            "shaft", "hoistway", "cabin", "car operating panel",
        ],
    },
    {
        "scope": "cabin",
        "triggers": [
            r"\bin\s+the\s+(?:cabin|car)\b",
            r"\bin[-\s]?car\b",
            r"\bon\s+the\s+COP\b",
            r"\bcar\s+operating\s+panel\b",
        ],
        "keywords": [
            "cabin", "car", "COP", "car operating panel", "buttons",
            "door operator", "Fermator", "GEZE", "glass door",
            "indicator", "DNXS",
        ],
        "penalise": [
            "controller cabinet", "machine room",
        ],
    },
    {
        "scope": "landing",
        "triggers": [
            r"\bat\s+the\s+landing\b",
            r"\bon\s+the\s+landing\s+door\b",
            r"\blanding\s+lock\b",
        ],
        "keywords": [
            "landing", "landing door", "landing lock", "interlock",
            "Fermator landing", "swing door",
        ],
        "penalise": [
            "cabin", "car operating panel",
        ],
    },
    {
        "scope": "selector_card",
        "triggers": [
            r"\bACS\s+selector\b",
            r"\bSMS\s+selector\b",
            r"\bselector\s+card\b",
            r"\bon\s+the\s+ACS\b",
            r"\bon\s+the\s+SMS\b",
        ],
        "keywords": [
            "ACS", "SMS", "selector", "DIP switch", "DL1", "DL2", "DL3", "DL4",
            "IN1", "IN2", "binary", "pin assignment", "jumper J3",
            "101.06.E1SEL", "expansion card",
        ],
        "penalise": [
            "main controller", "drive card",
        ],
    },
]


def detect_scope(query: str) -> Optional[Dict]:
    """
    Detect a scope qualifier in the user's query.

    Returns the matching scope definition dict (with `scope`, `keywords`, `penalise`)
    or None if no scope detected. If multiple scopes match, returns the first
    in SCOPE_DEFINITIONS order (most specific first).
    """
    if not query:
        return None
    q = query.lower()
    for sdef in SCOPE_DEFINITIONS:
        for trigger in sdef["triggers"]:
            if re.search(trigger, q, re.IGNORECASE):
                return {
                    "scope": sdef["scope"],
                    "keywords": list(sdef["keywords"]),
                    "penalise": list(sdef["penalise"]),
                    "matched_trigger": trigger,
                }
    return None


def scope_keywords(query: str) -> List[str]:
    """Convenience: just the keyword list, or [] if no scope detected."""
    s = detect_scope(query)
    return s["keywords"] if s else []


def scope_penalty_terms(query: str) -> List[str]:
    """Convenience: just the penalise list, or [] if no scope detected."""
    s = detect_scope(query)
    return s["penalise"] if s else []


def expand_query_for_scope(query: str, current_keywords: List[str]) -> List[str]:
    """
    Given the user query and the keywords already produced by the slang
    interceptor, append scope-specific keywords (deduped, case-insensitive).

    Returns the expanded keyword list.
    """
    extra = scope_keywords(query)
    if not extra:
        return current_keywords

    existing = {k.lower() for k in current_keywords}
    expanded = list(current_keywords)
    for k in extra:
        if k.lower() not in existing:
            expanded.append(k)
            existing.add(k.lower())
    return expanded


def apply_scope_penalty(results: List[Dict], query: str, penalty: float = 0.5) -> List[Dict]:
    """
    Optional reranking helper. If a scope was detected, multiply the score of
    any result whose content contains a 'penalise' term (and DOES NOT contain
    a scope keyword) by `penalty`. Mutates and returns the list.

    Expects each result to have either 'rrf_score' or 'similarity'. Quietly
    no-ops if neither is present.
    """
    s = detect_scope(query)
    if not s:
        return results

    pen = [t.lower() for t in s["penalise"]]
    boost = [k.lower() for k in s["keywords"]]
    if not pen and not boost:
        return results

    for r in results:
        content = (r.get("content") or "").lower()
        score_field = "rrf_score" if "rrf_score" in r else ("similarity" if "similarity" in r else None)
        if not score_field:
            continue
        # Hits a penalty term but NOT a scope keyword -> demote
        has_penalty = any(p in content for p in pen)
        has_boost = any(b in content for b in boost)
        if has_penalty and not has_boost:
            r[score_field] = r[score_field] * penalty

    # Resort by the score field
    if results:
        sf = "rrf_score" if "rrf_score" in results[0] else ("similarity" if "similarity" in results[0] else None)
        if sf:
            results.sort(key=lambda x: x.get(sf, 0), reverse=True)

    return results
