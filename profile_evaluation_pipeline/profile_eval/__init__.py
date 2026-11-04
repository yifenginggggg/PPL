"""Public fact-block profile evaluation pipeline."""

from .backends import Backend, BackendError, ClaudeCLIBackend, CommandBackend, ModelReply
from .blackbox import (
    LLMClient,
    PRIMARY_SCORE_NAMES,
    SCORING_VERSION,
    build_final_request,
    evaluate_batch,
    evaluate_final,
)
from .contract import PROTOCOL_VERSION, PUBLIC_LABELS, PUBLIC_TASKS, STAGE_PROTOCOL_VERSION
from .fact_pipeline import FACT_PROTOCOL_VERSION, evaluate_public_fact_graph

__all__ = [
    "Backend", "BackendError", "ClaudeCLIBackend", "CommandBackend", "ModelReply",
    "LLMClient", "PRIMARY_SCORE_NAMES", "SCORING_VERSION",
    "build_final_request", "evaluate_final", "evaluate_batch", "evaluate_public_fact_graph",
    "FACT_PROTOCOL_VERSION",
    "PROTOCOL_VERSION", "STAGE_PROTOCOL_VERSION", "PUBLIC_LABELS", "PUBLIC_TASKS",
]
