"""FactBlock validation, operation compilation and three-metric scoring."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from difflib import SequenceMatcher
import re
from typing import Any


FACT_PROTOCOL_VERSION = "fact-block-preprocessing-1.0"
FACT_GRAPH_FIELDS = {
    "source_blocks", "profile_blocks", "events", "coverage", "uncertainty",
}
FACT_BLOCK_FIELDS = {
    "fact_id", "origin", "stage", "profile_id", "unit_id", "text", "kind",
    "owner", "evidence_strength", "importance", "state", "assertion_status",
    "condition_type", "condition", "evidence", "links", "uncertainty",
}
FACT_EVENT_FIELDS = {
    "event_id", "event_type", "stage", "target_fact_id", "before_fact_id",
    "after_fact_id", "certainty", "scope", "evidence", "uncertainty",
}
FACT_RELATIONS = {
    "equivalent", "partial", "contradictory", "unsupported", "overgeneralized",
    "stale", "uncertain",
}
SEVERE_PROFILE_RELATIONS = {
    "unsupported", "contradictory", "overgeneralized", "stale", "uncertain",
}
FACT_STATES = {"active", "withdrawn", "superseded", "uncertain"}
EVENT_TYPES = {"add", "update", "withdraw", "reintroduce"}
OWNER_VALUES = {
    "user", "third_party", "authored_payload", "fictional_or_roleplay",
    "mixed", "uncertain", None,
}
IMPORTANCE_VALUES = {"required", "optional", "excluded", "uncertain"}
CONDITION_TYPES = {"decision", "context", "none", "uncertain"}
ASSERTION_STATUS = {"current", "historical", "quoted", "control", "uncertain"}

WITHDRAWAL_REQUEST_RE = re.compile(
    r"(?:^|[.!?]\s+|\b(?:please|kindly|can you|could you|would you|will you|"
    r"i want you to|i need you to|i'd like you to)\b[,\s]+)"
    r"(?:please\b[,\s]+)?"
    r"(?:forget\b|"
    r"remove\s+[^.?!]*\bfrom\s+(?:your|my|the)\s+memory\b|"
    r"delete\s+[^.?!]*\bfrom\s+(?:your|my|the)\s+memory\b|"
    r"stop\s+recommending\b|no\s+longer\s+recommend\b)",
    flags=re.IGNORECASE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_evidence(row: Mapping[str, Any], origin: str) -> None:
    _require(isinstance(row, Mapping), "evidence must be an object")
    _require(row.get("kind") == origin, f"{origin} evidence kind mismatch")
    if origin == "source":
        _require(isinstance(row.get("sid"), str), "source evidence needs sid")
        _require(isinstance(row.get("stage"), int), "source evidence needs stage")
        _require(isinstance(row.get("quote"), str), "source evidence needs quote")
    else:
        for field in ("profile_id", "unit_id", "span"):
            _require(isinstance(row.get(field), str), f"profile evidence needs {field}")


def validate_fact_graph(
    graph: Mapping[str, Any], request: Mapping[str, Any] | None = None
) -> dict[str, int]:
    """Validate the complete FactBlock/FactEvent graph and exact evidence."""
    _require(set(graph) == FACT_GRAPH_FIELDS, "fact graph field inventory mismatch")
    source = list(graph["source_blocks"])
    profile = list(graph["profile_blocks"])
    events = list(graph["events"])
    all_blocks = source + profile
    fact_ids = [row.get("fact_id") for row in all_blocks]
    _require(
        len(fact_ids) == len(set(fact_ids)) and all(isinstance(x, str) and x for x in fact_ids),
        "fact IDs must be unique",
    )
    source_ids = {row["fact_id"] for row in source}
    for row in all_blocks:
        _require(set(row) == FACT_BLOCK_FIELDS, "fact block field inventory mismatch")
        _require(row["origin"] in {"source", "profile"}, "invalid fact origin")
        _require(type(row["stage"]) is int and row["stage"] >= 1, "invalid fact stage")
        _require(isinstance(row["text"], str) and row["text"].strip(), "fact text required")
        _require(row["owner"] in OWNER_VALUES, "invalid fact owner")
        _require(row["evidence_strength"] in {"explicit", "grounded_inference", "authored_payload", "uncertain"}, "invalid evidence strength")
        _require(row["importance"] in IMPORTANCE_VALUES, "invalid importance")
        _require(row["state"] in FACT_STATES, "invalid fact state")
        _require(row["assertion_status"] in ASSERTION_STATUS, "invalid assertion status")
        _require(row["condition_type"] in CONDITION_TYPES, "invalid condition type")
        _require(row["condition"] is None or isinstance(row["condition"], str), "invalid condition")
        if row["origin"] == "source":
            _require(row["profile_id"] is None and row["unit_id"] is None, "source block cannot have profile identity")
        else:
            _require(isinstance(row["profile_id"], str) and isinstance(row["unit_id"], str), "profile block needs profile identity")
        _require(isinstance(row["evidence"], list), "fact evidence must be a list")
        for evidence in row["evidence"]:
            _validate_evidence(evidence, row["origin"])
        _require(isinstance(row["links"], list), "fact links must be a list")
        for link in row["links"]:
            _require(set(link) == {"fact_id", "relation"}, "fact link fields mismatch")
            _require(link["fact_id"] in source_ids, "profile link must reference a source block")
            _require(link["relation"] in FACT_RELATIONS, "invalid fact link relation")
    for event in events:
        _require(set(event) == FACT_EVENT_FIELDS, "fact event field inventory mismatch")
        _require(event["event_type"] in EVENT_TYPES, "invalid event type")
        _require(type(event["stage"]) is int and event["stage"] >= 1, "invalid event stage")
        for key in ("target_fact_id", "before_fact_id", "after_fact_id"):
            _require(event[key] is None or event[key] in fact_ids, f"invalid {key}")
        _require(event["certainty"] in {"clear", "uncertain"}, "invalid event certainty")
        _require(isinstance(event["scope"], str), "event scope required")
        _require(isinstance(event["evidence"], list), "event evidence must be a list")
        for evidence in event["evidence"]:
            _validate_evidence(evidence, "source")
    _require(isinstance(graph["coverage"], Mapping), "coverage must be an object")
    _require(isinstance(graph["uncertainty"], list), "uncertainty must be a list")

    if request is not None:
        messages = {row["sid"]: row for row in request["history"]["messages"]}
        profiles = {row["profile_id"]: row for row in request["profiles"]}
        supplied_ids = set(profiles)
        seen_ids = {row["profile_id"] for row in profile}
        _require(supplied_ids <= seen_ids, "profile block inventory does not cover requested profiles")

        def unit_text(profile_id: str, unit_id: str) -> str:
            profile_row = profiles.get(profile_id)
            if not profile_row:
                return ""
            for index, unit in enumerate(profile_row.get("units", []), 1):
                if isinstance(unit, Mapping):
                    candidate = str(unit.get("unit_id", unit.get("id", f"C{index}")))
                    if not candidate.startswith(profile_id + "/"):
                        candidate = f"{profile_id}/{candidate}"
                    if candidate == unit_id:
                        return str(unit.get("text", ""))
                elif isinstance(unit, list) and len(unit) == 4 and f"{profile_id}/C{unit[0]}" == unit_id:
                    return str(unit[3])
            return ""

        for row in all_blocks:
            for evidence in row["evidence"]:
                if row["origin"] == "source":
                    message = messages.get(evidence["sid"])
                    _require(message is not None, "source evidence SID is not in history")
                    _require(evidence["quote"] in message["text"], "source evidence is not an exact history substring")
                    _require(evidence["stage"] == message["stage"], "source evidence stage mismatch")
                else:
                    _require(evidence["profile_id"] == row["profile_id"], "profile evidence profile mismatch")
                    _require(evidence["unit_id"] == row["unit_id"], "profile evidence unit mismatch")
                    _require(evidence["span"] in unit_text(row["profile_id"], row["unit_id"]), "profile evidence is not an exact profile substring")
        for event in events:
            for evidence in event["evidence"]:
                message = messages.get(evidence["sid"])
                _require(message is not None, "event evidence SID is not in history")
                _require(evidence["quote"] in message["text"], "event evidence is not an exact history substring")
                _require(evidence["stage"] == message["stage"], "event evidence stage mismatch")

    return {
        "source_blocks_n": len(source),
        "profile_blocks_n": len(profile),
        "events_n": len(events),
        "uncertainty_n": len(graph["uncertainty"]),
    }


def _profile_groups(blocks: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        groups[(str(block["profile_id"]), str(block["unit_id"]))].append(block)
    return groups


def _is_withdrawal_request(text: Any) -> bool:
    return bool(WITHDRAWAL_REQUEST_RE.search(str(text or "").strip()))


def _withdrawal_scope(text: str) -> str:
    scope = " ".join(str(text).strip().split())
    scope = re.sub(r"^\s*(?:please\s+)?(?:forget|remove|delete)\s+", "", scope, flags=re.IGNORECASE)
    scope = re.sub(r"^\s*(?:that|the|my|this)\s+", "", scope, flags=re.IGNORECASE)
    scope = re.sub(r"\s+from\s+(?:your|my|the)\s+memory\s*[.!?]*$", "", scope, flags=re.IGNORECASE)
    return scope.strip(" \t\r\n.!?") or str(text).strip()


def _scope_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).lower()) if len(token) > 1}


def _scope_matches_block(scope: str, block: Mapping[str, Any]) -> bool:
    block_text = str(block.get("text", ""))
    if not block_text:
        return False
    scope_norm = " ".join(str(scope).lower().split())
    block_norm = " ".join(block_text.lower().split())
    if scope_norm and (scope_norm in block_norm or block_norm in scope_norm):
        return True
    scope_terms = _scope_terms(scope)
    block_terms = _scope_terms(block_text)
    overlap = len(scope_terms & block_terms)
    return len(scope_terms) >= 2 and overlap >= 2 and overlap / len(scope_terms) >= 0.6


def _withdrawal_similarity(scope: str, text: str) -> float:
    scope_terms = _scope_terms(scope)
    text_terms = _scope_terms(text)
    overlap = len(scope_terms & text_terms) / len(scope_terms) if scope_terms else 0.0
    return max(overlap, SequenceMatcher(None, scope.lower(), text.lower()).ratio())


def _derive_history_withdrawals(graph: Mapping[str, Any], request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile explicit history withdrawal requests into FactEvents."""
    messages = request.get("history", {}).get("messages", [])
    source = list(graph.get("source_blocks", []))
    source_by_id = {row.get("fact_id"): row for row in source}
    request_source_ids = {row.get("fact_id") for row in source if _is_withdrawal_request(row.get("text"))}
    events = [row for row in graph.get("events", []) if row.get("event_type") != "withdraw"]
    model_withdrawals = [row for row in graph.get("events", []) if row.get("event_type") == "withdraw"]
    output = []
    for index, message in enumerate(messages, 1):
        text = str(message.get("text", ""))
        if not _is_withdrawal_request(text):
            continue
        sid = str(message.get("sid", f"u{index}"))
        stage = int(message.get("stage", 1))
        scope = _withdrawal_scope(text)
        target = None
        for event in model_withdrawals:
            candidate = event.get("target_fact_id")
            candidate_row = source_by_id.get(candidate)
            if candidate in request_source_ids or candidate_row is None:
                continue
            if int(event.get("stage", stage)) == stage and _scope_matches_block(scope, candidate_row):
                target = candidate
                break
        if target is None:
            candidates = [row for row in source if row.get("fact_id") not in request_source_ids and int(row.get("stage", 1)) <= stage and row.get("state") != "uncertain"]
            ranked = sorted(candidates, key=lambda row: (row.get("state") == "active", int(row.get("stage", 1)) == stage, _withdrawal_similarity(scope, str(row.get("text", "")))), reverse=True)
            if ranked and _withdrawal_similarity(scope, str(ranked[0].get("text", ""))) >= 0.30:
                target = str(ranked[0]["fact_id"])
        output.append({
            "event_id": f"W_{sid}", "event_type": "withdraw", "stage": stage,
            "target_fact_id": target, "before_fact_id": target, "after_fact_id": None,
            "certainty": "clear", "scope": scope,
            "evidence": [{"kind": "source", "sid": sid, "stage": stage, "quote": text}],
            "uncertainty": None,
        })
    return events + output


def _effective_graph(graph: Mapping[str, Any], request: Mapping[str, Any] | None) -> dict[str, Any]:
    effective = dict(graph)
    if request is not None:
        effective["events"] = _derive_history_withdrawals(graph, request)
    return effective


def compile_public_materials(graph: Mapping[str, Any]) -> dict[str, Any]:
    validate_fact_graph(graph)
    source = [row for row in graph["source_blocks"] if row["origin"] == "source"]
    profile = [row for row in graph["profile_blocks"] if row["origin"] == "profile"]
    groups = _profile_groups(profile)
    profile_claims = [
        {
            "profile_id": profile_id,
            "unit_id": unit_id,
            "blocks": blocks,
            "coverage": graph["coverage"].get("unit_coverage", {}).get(unit_id),
        }
        for (profile_id, unit_id), blocks in sorted(groups.items())
    ]
    uncertain = list(graph["uncertainty"])
    uncertain.extend({"kind": "fact", "fact_id": row["fact_id"], "reason": row["uncertainty"]} for row in source + profile if row["uncertainty"])
    uncertain.extend({"kind": "link", "fact_id": row["fact_id"], "link": link} for row in profile for link in row["links"] if link["relation"] == "uncertain")
    return {
        "source_catalogue": {"source_blocks": source, "events": list(graph["events"])},
        "profile_claims": profile_claims,
        "uncertain": uncertain,
    }


def _evaluate_completeness(source: dict[str, dict[str, Any]], groups: Mapping) -> dict[str, Any]:
    items = []
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (profile_id, _), blocks in groups.items():
        by_profile[profile_id].extend(blocks)
    for profile_id, blocks in by_profile.items():
        stage = max(block["stage"] for block in blocks)
        for fact_id, source_row in source.items():
            if source_row["stage"] > stage or source_row["state"] not in {"active", "uncertain"}:
                continue
            relations = {link["relation"] for block in blocks for link in block["links"] if link["fact_id"] == fact_id}
            label = "U" if "uncertain" in relations else "D" if "contradictory" in relations else "P" if "partial" in relations else "C" if "equivalent" in relations else "M"
            items.append({"profile_id": profile_id, "source_fact_id": fact_id, "label": label})
    return {"items": items, "items_n": len(items)}


def _evaluate_support(groups: Mapping) -> dict[str, Any]:
    items = []
    for (profile_id, unit_id), blocks in groups.items():
        if any(block["assertion_status"] == "control" for block in blocks):
            label = "C"
        else:
            relations = [link["relation"] for block in blocks for link in block["links"]]
            if any(value == "uncertain" for value in relations):
                label = "U"
            elif any(value == "overgeneralized" for value in relations):
                label = "G"
            elif any(value == "stale" for value in relations):
                label = "V"
            elif any(value == "unsupported" for value in relations):
                label = "X" if any(value == "equivalent" for value in relations) else "A"
            elif relations and all(value == "equivalent" for value in relations):
                label = "S"
            else:
                label = "U"
        items.append({"profile_id": profile_id, "unit_id": unit_id, "label": label})
    return {"items": items, "items_n": len(items)}


def _evaluate_reliability(groups: Mapping) -> dict[str, Any]:
    items = []
    for blocks in groups.values():
        for block in blocks:
            if block.get("assertion_status") == "control":
                continue
            relations = {link["relation"] for link in block.get("links", [])}
            clean = bool(relations) and block.get("assertion_status") != "uncertain" and not relations.intersection(SEVERE_PROFILE_RELATIONS)
            items.append({"profile_id": block["profile_id"], "unit_id": block["unit_id"], "fact_id": block["fact_id"], "label": "clean" if clean else "error"})
    return {"items": items, "items_n": len(items)}


def evaluate_public_fact_graph(graph: Mapping[str, Any], request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the fixed preprocessing and Completeness/Support/Reliability tasks."""
    effective_graph = _effective_graph(graph, request)
    validate_fact_graph(effective_graph)
    source = {row["fact_id"]: row for row in effective_graph["source_blocks"]}
    groups = _profile_groups(effective_graph["profile_blocks"])
    return {
        "materials": compile_public_materials(effective_graph),
        "tasks": {
            "completeness": _evaluate_completeness(source, groups),
            "support": _evaluate_support(groups),
            "reliability": _evaluate_reliability(groups),
        },
    }
