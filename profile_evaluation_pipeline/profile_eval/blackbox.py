"""Public black-box API for profile evaluation.

Callers provide a history, one profile and a model client.  The core does not
read persona files or impose a stage scope.  The
model is called twice: history-only source extraction, then profile projection
against the immutable source blocks. Python computes the three fixed headline
metrics: Completeness, Support and Reliability.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Protocol

from .backends import ModelReply
from .fact_pipeline import (
    FACT_PROTOCOL_VERSION,
    evaluate_public_fact_graph,
    validate_fact_graph,
)


SCORING_VERSION = "scoring-v2.3-three-headline"
PRIMARY_SCORE_NAMES = (
    "completeness", "support", "reliability",
)

# Relation weights are used only for the source-balanced headline rates.  A
# partial link receives half credit; unsupported, stale, contradictory and
# uncertain links receive no credit.  This keeps the source denominator shared
# across profiles while still preserving the original relation labels in the
# fact graph.
_FAIR_RELATION_WEIGHTS = {
    "equivalent": 1.0,
    "partial": 0.5,
    "contradictory": 0.0,
    "unsupported": 0.0,
    "overgeneralized": 0.0,
    "stale": 0.0,
    "uncertain": 0.0,
}


class LLMClient(Protocol):
    """Provider-neutral client contract used by the black box."""

    def generate(self, *, system_prompt: str, payload: dict[str, Any],
                 model_id: str) -> Any:
        ...


def _history_messages(history: Any) -> list[dict[str, Any]]:
    if isinstance(history, Mapping):
        messages = history.get("messages")
    else:
        messages = history
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("history must be a message list or {messages: [...]}")
    output = []
    current_stage = 1
    for index, raw in enumerate(messages, 1):
        if isinstance(raw, str):
            raw = {"text": raw}
        if not isinstance(raw, Mapping):
            raise ValueError(f"history message {index} must be an object or string")
        role = raw.get("role", "user")
        if role is not None and (
            not isinstance(role, str) or role.strip().lower() != "user"
        ):
            continue
        text = raw.get("text", raw.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"history message {index} needs nonempty text")
        sid = raw.get("sid", raw.get("id", f"u{index}"))
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"history message {index} needs a string id")
        stage = raw.get("stage", current_stage)
        if type(stage) is not int or stage < 1:
            raise ValueError(f"history message {sid} has invalid stage")
        output.append({"sid": sid, "stage": stage, "text": text})
        current_stage = max(current_stage, stage)
    if not output:
        raise ValueError("history must contain at least one user message")
    if len({row["sid"] for row in output}) != len(output):
        raise ValueError("history message ids must be unique")
    return output


def _profile_input(profile: Any) -> dict[str, Any]:
    if isinstance(profile, str):
        profile = {"text": profile}
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a string or object")
    profile_id = profile.get("profile_id", profile.get("id", "profile"))
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile needs profile_id or id")
    stage = profile.get("stage", 8)
    if type(stage) is not int or stage < 1:
        raise ValueError("profile stage must be a positive integer")
    text = profile.get("text", "")
    units = profile.get("units")
    if units is None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("profile needs nonempty text or units")
        units = [{"unit_id": f"{profile_id}/C1", "text": text, "cid": 1}]
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ValueError("profile units must be a list")
    normalized_units = []
    for index, raw in enumerate(units, 1):
        if isinstance(raw, str):
            raw = {"text": raw}
        if not isinstance(raw, Mapping):
            raise ValueError(f"profile unit {index} must be an object or string")
        unit_id = raw.get("unit_id", raw.get("id", f"{profile_id}/C{index}"))
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError(f"profile unit {index} needs unit_id")
        if not unit_id.startswith(profile_id + "/"):
            unit_id = f"{profile_id}/{unit_id}"
        unit_text = raw.get("text", raw.get("content", ""))
        if not isinstance(unit_text, str):
            raise ValueError(f"profile unit {unit_id} needs text")
        normalized_units.append({
            "unit_id": unit_id,
            "text": unit_text,
            "cid": int(raw.get("cid", index)),
        })
    full_text = text if isinstance(text, str) else "\n".join(
        unit["text"] for unit in normalized_units
    )
    return {
        "profile_id": profile_id,
        "stage": stage,
        "text": full_text,
        "units": normalized_units,
        "visibility": {
            "profile_stage": stage,
            "history_scope": "full",
        },
    }


def build_final_request(*, history: Any, profile: Any,
                        model_id: str = "model") -> dict[str, Any]:
    """Build the normalized context used by the two extraction stages."""
    messages = _history_messages(history)
    normalized_profile = _profile_input(profile)
    return {
        "protocol_version": FACT_PROTOCOL_VERSION,
        "model_id": model_id,
        "history": {
            "messages": messages,
            "profile_visibility": {
                normalized_profile["profile_id"]: normalized_profile["visibility"]
            },
        },
        "profiles": [normalized_profile],
        "profile_mode": "final",
        "output_schema": {
            "history_source": "source_blocks[]",
            "profile_projection": "profile_blocks[] + coverage + uncertainty",
        },
    }


def _call_llm(llm: Any, *, payload: dict[str, Any], model_id: str,
              system_prompt: str) -> tuple[Any, dict[str, Any]]:
    if hasattr(llm, "generate"):
        reply = llm.generate(system_prompt=system_prompt,
                             payload=payload, model_id=model_id)
    elif callable(llm):
        reply = llm(system_prompt=system_prompt,
                    payload=payload, model_id=model_id)
    else:
        raise TypeError("llm must provide generate(...) or be callable")
    metadata: dict[str, Any] = {}
    if isinstance(reply, ModelReply):
        return reply.text, dict(reply.metadata)
    if isinstance(reply, Mapping):
        return reply, metadata
    if isinstance(reply, str):
        return reply, metadata
    raise TypeError("llm response must be ModelReply, JSON object, or string")


def _parse_graph(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        raise ValueError("LLM response must be JSON object or JSON string")
    try:
        value = json.loads(raw)
        if isinstance(value, str):
            value = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        raise ValueError("LLM response is not valid JSON") from None
    if not isinstance(value, Mapping):
        raise ValueError("LLM response JSON must be an object")
    return value


def _repair_evidence(graph: dict[str, Any], request: Mapping[str, Any]) -> list[str]:
    """Make malformed provider evidence auditable without treating it as valid.

    The provider may paraphrase a profile span while still returning a useful
    semantic link.  We replace only the citation with the exact caller-supplied
    span and downgrade every affected link to ``uncertain``.  No text is
    invented, and the warning is returned in the result envelope.
    """
    messages = {row["sid"]: row for row in request["history"]["messages"]}
    profiles = {row["profile_id"]: row for row in request["profiles"]}
    unit_texts: dict[tuple[str, str], str] = {}
    for profile_id, profile in profiles.items():
        for index, unit in enumerate(profile.get("units", []), 1):
            if isinstance(unit, Mapping):
                unit_id = str(unit.get("unit_id", unit.get("id", f"C{index}")))
                text = str(unit.get("text", ""))
            elif isinstance(unit, list) and len(unit) == 4:
                unit_id, text = f"{profile_id}/C{unit[0]}", str(unit[3])
            else:
                unit_id, text = f"{profile_id}/C{index}", str(unit)
            if not unit_id.startswith(profile_id + "/"):
                unit_id = f"{profile_id}/{unit_id}"
            unit_texts[(profile_id, unit_id)] = text

    warnings: list[str] = []
    for block in graph.get("source_blocks", []) + graph.get("profile_blocks", []):
        repaired = False
        unresolved = False
        if block.get("origin") == "source":
            for evidence in block.get("evidence", []):
                message = messages.get(evidence.get("sid"))
                if message and (not isinstance(evidence.get("quote"), str)
                                or evidence.get("quote") not in message.get("text", "")):
                    evidence["quote"] = message["text"]
                    repaired = True
                elif not message:
                    unresolved = True
        else:
            profile_id = str(block.get("profile_id"))
            unit_id = str(block.get("unit_id"))
            exact = unit_texts.get((profile_id, unit_id), "")
            for evidence in block.get("evidence", []):
                if exact and (not isinstance(evidence.get("span"), str)
                              or evidence.get("span") not in exact):
                    evidence["span"] = exact
                    repaired = True
                elif not exact:
                    unresolved = True
        if repaired:
            block["evidence_strength"] = "grounded_inference"
            for link in block.get("links", []):
                link["relation"] = "uncertain"
            warnings.append(f"{block.get('fact_id')}: provider evidence repaired from exact input")
        if unresolved:
            reason = "provider evidence could not be matched to supplied input"
            block["uncertainty"] = f"{block.get('uncertainty')}; {reason}" if block.get("uncertainty") else reason
            for link in block.get("links", []):
                link["relation"] = "uncertain"
            warnings.append(f"{block.get('fact_id')}: {reason}")
    return warnings


def _fair_relation_scores(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Compute profile-balanced claim precision and source recall.

    The old headline scores used profile-local block denominators for every
    task. A profile that emitted fewer claims could therefore look better
    even when it omitted more history facts.  This helper uses one common
    source-fact denominator for recall and treats unlinked/unsupported profile
    blocks as precision failures.
    """
    source = {row["fact_id"]: row for row in graph.get("source_blocks", [])}
    profile_blocks = [row for row in graph.get("profile_blocks", [])
                      if row.get("assertion_status") != "control"]
    if not profile_blocks:
        return {
            "source_recall": None, "source_exact_recall": None,
            "profile_precision": None, "profile_exact_precision": None,
            "grounded_f1": None, "eligible_source_n": 0,
            "profile_blocks_n": 0,
        }
    stage = max(int(row.get("stage", 1)) for row in profile_blocks)
    eligible = {
        fact_id for fact_id, row in source.items()
        if int(row.get("stage", 1)) <= stage
        and row.get("state") == "active"
        and row.get("owner") == "user"
        and row.get("importance") == "required"
    }
    best_by_source = {fact_id: 0.0 for fact_id in eligible}
    profile_values = []
    exact_profile_n = 0
    for block in profile_blocks:
        relations = [link.get("relation") for link in block.get("links", [])
                     if link.get("fact_id") in source]
        weights = [_FAIR_RELATION_WEIGHTS.get(value, 0.0) for value in relations]
        block_value = min(weights) if weights else 0.0
        profile_values.append(block_value)
        if relations and all(value == "equivalent" for value in relations):
            exact_profile_n += 1
        for link in block.get("links", []):
            fact_id = link.get("fact_id")
            if fact_id in best_by_source:
                best_by_source[fact_id] = max(
                    best_by_source[fact_id],
                    _FAIR_RELATION_WEIGHTS.get(link.get("relation"), 0.0),
                )
    source_recall = (sum(best_by_source.values()) / len(best_by_source)
                     if best_by_source else None)
    source_exact_recall = (sum(value == 1.0 for value in best_by_source.values())
                           / len(best_by_source) if best_by_source else None)
    profile_precision = sum(profile_values) / len(profile_values)
    profile_exact_precision = exact_profile_n / len(profile_blocks)
    if source_recall is None or profile_precision is None:
        grounded_f1 = None
    elif source_recall + profile_precision > 0:
        grounded_f1 = (
            2 * source_recall * profile_precision
            / (source_recall + profile_precision)
        )
    else:
        grounded_f1 = 0.0
    return {
        "source_recall": source_recall,
        "source_exact_recall": source_exact_recall,
        "profile_precision": profile_precision,
        "profile_exact_precision": profile_exact_precision,
        "grounded_f1": grounded_f1,
        "eligible_source_n": len(eligible),
        "profile_blocks_n": len(profile_blocks),
    }


def _fair_metric(rate: float | None, *, eligible_n: int,
                 exact_rate: float | None = None) -> dict[str, Any]:
    return {
        "pass_n": None,
        "eligible_n": eligible_n,
        "resolved_n": eligible_n,
        "uncertain_n": 0,
        "rate": rate,
        "conditional_rate": rate,
        "exact_rate": exact_rate,
    }


def _final_scores(graph: Mapping[str, Any], tasks: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed three-metric public score vector.

    Completeness is source-fact recall, Support is grounded F1 (source recall
    plus profile precision), and Reliability is the rate of profile blocks
    without a severe semantic or provenance error.
    """
    fair = _fair_relation_scores(graph)
    profile_id = str(graph["profile_blocks"][0]["profile_id"])

    reliability_items = list(tasks.get("reliability", {}).get("items", []))
    reliability_pass = sum(item.get("label") == "clean" for item in reliability_items)
    reliability = {
        "pass_n": reliability_pass,
        "eligible_n": len(reliability_items),
        "resolved_n": len(reliability_items),
        "uncertain_n": 0,
        "rate": reliability_pass / len(reliability_items) if reliability_items else None,
        "conditional_rate": reliability_pass / len(reliability_items) if reliability_items else None,
    }

    # Completeness is the strict source-fact recall used by the original C/P/M
    # contract.  Weighted source recall is retained in fair_detail as an
    # auxiliary coverage diagnostic; otherwise many partial links can make a
    # weakly linked profile look complete.
    primary = {
        "completeness": _fair_metric(
            fair["source_exact_recall"], eligible_n=fair["eligible_source_n"],
            exact_rate=fair["source_exact_recall"],
        ),
        "support": _fair_metric(
            fair["grounded_f1"], eligible_n=fair["profile_blocks_n"],
            exact_rate=fair["profile_exact_precision"],
        ),
        "reliability": reliability,
    }
    result = {
        "profile_id": profile_id,
        "primary": primary,
        "fair_detail": fair,
        **primary,
        "scoring_version": SCORING_VERSION,
    }
    primary["completeness"]["weighted_rate"] = fair["source_recall"]
    return result


def evaluate_final(*, history: Any, profile: Any, llm: Any,
                   model_id: str = "model", max_attempts: int = 2) -> dict[str, Any]:
    """Evaluate one profile without filesystem or provider assumptions.

    One successful evaluation uses two model calls: one history-only source
    call and one source-plus-profile projection call. Retries rerun the failed
    attempt with the same two-stage protocol.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    request = build_final_request(history=history, profile=profile, model_id=model_id)
    from .two_stage import (
        HISTORY_STAGE_PROMPT,
        PROFILE_STAGE_PROMPT,
        build_history_payload,
        build_profile_payload,
        finalize_two_stage_graph,
        normalize_source_stage,
    )

    profile_id = request["profiles"][0]["profile_id"]
    history_payload, history_context = build_history_payload(
        persona_id=profile_id, history=history, model_id=model_id)
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            source_raw, source_metadata = _call_llm(
                llm, payload=history_payload, model_id=model_id,
                system_prompt=HISTORY_STAGE_PROMPT,
            )
            source_blocks, source_warnings = normalize_source_stage(
                _parse_graph(source_raw), history_context)
            profile_payload = build_profile_payload(
                persona_id=profile_id, request=request, source_blocks=source_blocks)
            profile_raw, profile_metadata = _call_llm(
                llm, payload=profile_payload, model_id=model_id,
                system_prompt=PROFILE_STAGE_PROMPT,
            )
            graph, profile_warnings = finalize_two_stage_graph(
                source_blocks, _parse_graph(profile_raw), request)
            warnings = source_warnings + profile_warnings + _repair_evidence(graph, request)
            validation = validate_fact_graph(graph, request=request)
            compiled = evaluate_public_fact_graph(graph)
            metadata = {
                "history": source_metadata,
                "profile": profile_metadata,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
            break
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
    else:
        raise last_error or ValueError("black-box evaluation failed")
    return {
        "protocol_version": FACT_PROTOCOL_VERSION,
        "mode": "final",
        "model_id": model_id,
        "provider_metadata": metadata,
        "normalization_warnings": warnings,
        "request": request,
        "fact_graph": graph,
        "materials": compiled["materials"],
        "tasks": compiled["tasks"],
        "scores": _final_scores(graph, compiled["tasks"]),
        "validation": validation,
    }


def evaluate_batch(*, history: Any, profiles: Sequence[Any], llm: Any,
                   model_id: str = "model", max_attempts: int = 2,
                   persona_id: str = "persona") -> dict[str, Any]:
    """Evaluate multiple profiles with one shared history extraction.

    The profile projections remain independent calls and retain the exact same
    result envelope as :func:`evaluate_final`. This is the batch orchestration
    used when several profiles belong to one history.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if isinstance(profiles, (str, bytes)) or not isinstance(profiles, Sequence):
        raise ValueError("profiles must be a nonempty sequence")
    if not profiles:
        raise ValueError("profiles must be a nonempty sequence")
    requests = [build_final_request(history=history, profile=profile,
                                    model_id=model_id)
                for profile in profiles]
    profile_ids = [request["profiles"][0]["profile_id"] for request in requests]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("profile ids must be unique within a batch")

    from copy import deepcopy
    from .two_stage import (
        HISTORY_STAGE_PROMPT,
        PROFILE_STAGE_PROMPT,
        build_history_payload,
        build_profile_payload,
        finalize_two_stage_graph,
        normalize_source_stage,
    )

    history_payload, history_context = build_history_payload(
        persona_id=persona_id, history=history, model_id=model_id)
    source_last_error = None
    source_blocks = None
    source_warnings: list[str] = []
    source_metadata: dict[str, Any] = {}
    source_attempt = 0
    for attempt in range(1, max_attempts + 1):
        try:
            source_raw, source_metadata = _call_llm(
                llm, payload=history_payload, model_id=model_id,
                system_prompt=HISTORY_STAGE_PROMPT,
            )
            source_blocks, source_warnings = normalize_source_stage(
                _parse_graph(source_raw), history_context)
            source_attempt = attempt
            break
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            source_last_error = exc
            if attempt == max_attempts:
                raise
    if source_blocks is None:
        raise source_last_error or ValueError("history source extraction failed")

    results: list[dict[str, Any]] = []
    for request in requests:
        profile_id = request["profiles"][0]["profile_id"]
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                profile_source = deepcopy(source_blocks)
                profile_payload = build_profile_payload(
                    persona_id=persona_id, request=request,
                    source_blocks=profile_source)
                profile_raw, profile_metadata = _call_llm(
                    llm, payload=profile_payload, model_id=model_id,
                    system_prompt=PROFILE_STAGE_PROMPT,
                )
                graph, profile_warnings = finalize_two_stage_graph(
                    profile_source, _parse_graph(profile_raw), request)
                warnings = source_warnings + profile_warnings + _repair_evidence(graph, request)
                validation = validate_fact_graph(graph, request=request)
                compiled = evaluate_public_fact_graph(graph)
                results.append({
                    "protocol_version": FACT_PROTOCOL_VERSION,
                    "mode": "final",
                    "model_id": model_id,
                    "provider_metadata": {
                        "history": source_metadata,
                        "profile": profile_metadata,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "shared_history": True,
                        "history_attempt": source_attempt,
                    },
                    "normalization_warnings": warnings,
                    "request": request,
                    "fact_graph": graph,
                    "materials": compiled["materials"],
                    "tasks": compiled["tasks"],
                    "scores": _final_scores(graph, compiled["tasks"]),
                    "validation": validation,
                })
                break
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    raise
        else:
            raise last_error or ValueError(
                f"profile projection failed for {profile_id}")

    return {
        "protocol_version": FACT_PROTOCOL_VERSION,
        "mode": "batch",
        "model_id": model_id,
        "source_catalogue": {
            "source_blocks": source_blocks,
            "normalization_warnings": source_warnings,
            "provider_metadata": source_metadata,
            "attempt": source_attempt,
        },
        "results": results,
    }


__all__ = ["LLMClient", "PRIMARY_SCORE_NAMES", "SCORING_VERSION",
           "build_final_request", "evaluate_final", "evaluate_batch"]
