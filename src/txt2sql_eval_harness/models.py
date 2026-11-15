"""Core data shapes and configuration models for the Text2SQL Decision-Model Judge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

Cell = None | int | float | str | bytes

ErrorKind = Literal[
    "compile_error",
    "non_read",
    "timeout",
    "fetch_cap",
    "multi_statement",
    "runtime",
    "resource_limit",
    "volatile",
]

MethodName = Literal[
    "B1",
    "B2",
    "B-IBM-subset",
    "B-IBM-judge",
    "B4",
    "B6",
    "B6+E",
    "H",
    "H+E",
]

RecordedMethodName = Literal[
    "B1",
    "B2",
    "B-IBM-subset",
    "B-IBM-judge",
    "B4",
    "B6",
    "B6+E",
    "H+E",
]

ModeName = Literal["sql_only", "full_trace"]
VerdictLabel = Literal["CORRECT", "INCORRECT", "AMBIGUOUS", "ABSTAIN"]
DecidedBy = Literal[
    "rule:D0",
    "rule:D1",
    "rule:D2",
    "rule:D3",
    "rule:D4",
    "rule:D5",
    "jev",
    "escalation",
    "baseline",
    "abstain",
]

COUNTEREXAMPLE_ERROR_KINDS: frozenset[str] = frozenset({"compile_error", "non_read"})
EXEC_UNAVAILABLE_ERROR_KINDS: frozenset[str] = frozenset(
    {"timeout", "runtime", "multi_statement", "resource_limit", "volatile"}
)


@dataclass
class ResultTable:
    """Result set returned by sandboxed SQL execution."""

    columns: list[str]
    rows: list[tuple[Cell, ...]]
    total_rows: int | None
    complete: bool
    cut_by: str | None = None


@dataclass
class Step:
    """One SQL execution step inside a Trace."""

    index: int
    sql: str
    result: ResultTable | None = None
    error_kind: ErrorKind | None = None


@dataclass
class Trace:
    """Agent or single-SQL Text2SQL trace."""

    trace_id: str
    group_id: str
    question: str
    hint: str | None
    db_id: str
    db_sha256: str
    steps: list[Step]
    final_answer: str | None = None


@dataclass
class GroundTruth:
    """Ground-truth reference query chain and optional alternative/answer."""

    item_ref: str
    sql_chain: list[str]
    alternatives: list[list[str]] = field(default_factory=list)
    reference_answer: str | None = None
    results: list[ResultTable] | None = None


@dataclass
class Contract:
    """Must-match contract derived from the question and ground-truth SQL."""

    gt_columns: list[int]
    order_required: bool
    limit_required: bool


@dataclass
class Alignment:
    """Column and row alignment between one GT result and one generated/agent result."""

    column_map: dict[int, int | None]
    map_status: Literal["unique", "resolved_by_search", "partial", "none", "unknown"]
    relation: Literal["equal", "gt_subset_gen", "gen_subset_gt", "overlap", "disjoint", "unknown"]
    key_columns: list[int] | None
    witness_kind: Literal["key_aligned", "unmatched_rows", "unmapped_rows", "none"]
    rows_matched: int
    rows_only_gt: int
    rows_only_gen: int
    order_ok: bool | None
    scale_match: float | None = None


@dataclass
class Derivation:
    """Code-computed column aggregate over a complete result table."""

    pool: Literal["reference", "agent"]
    step: int | None
    op: Literal["SUM", "AVG", "MIN", "MAX", "COUNT", "ROW_COUNT"]
    column: str | None
    value: Cell


@dataclass
class Claim:
    """Extracted atomic claim in a final answer and its bindings to result cells."""

    text: str
    kind: Literal["number", "date", "entity"]
    value: Decimal | str
    decimals: int | None
    in_question: bool
    gt_cells: list[tuple[int, int]] = field(default_factory=list)
    agent_cells: list[tuple[int, int, int]] = field(default_factory=list)
    derivations: list[Derivation] = field(default_factory=list)


@dataclass
class EvidencePacket:
    """Bounded, escaped evidence state rendered for model consumption."""

    tier: Literal["jev", "escalation"]
    state: str
    n_cells: int
    n_tokens: int
    required_met: bool
    truncated: list[str]
    packet_sha256: str


@dataclass
class Attempt:
    """Single HTTP/SDK attempt inside a ModelCall."""

    n: int
    wait_ms: int
    elapsed_ms: int
    result: Literal[
        "ok",
        "invalid",
        "http_4xx",
        "http_429",
        "http_5xx",
        "timeout",
        "connection_lost",
        "connect_refused",
    ]
    input_tokens: int
    output_tokens: int | None
    billing: Literal["reported", "estimated", "ambiguous", "none"]
    cost_usd: float


@dataclass
class CallMeasure:
    """End-to-end measurement of a ModelCall including retries and waits."""

    attempts: list[Attempt]
    wall_ms: int
    billed_usd: float
    ambiguous_usd: float


@dataclass
class ModelCall:
    """Recorded model invocation across Jev, generative escalation, baselines, or agent."""

    stage: Literal["jev", "escalation", "b4", "b6", "ibm_judge", "agent", "prompt_check"]
    model: str
    request_sha256: str
    questions: list[str]
    answers: dict[str, Any] | None
    outcome: Literal[
        "ok",
        "backend_error",
        "invalid_response",
        "spend_cap",
        "remote_disabled",
        "cache_miss",
    ]
    invalid_reason: str | None
    from_cache: bool
    cold: CallMeasure


@dataclass
class Verdict:
    """Final judgement produced by a method on a Trace."""

    trace_id: str
    method: MethodName
    mode: ModeName
    label: VerdictLabel
    decided_by: DecidedBy
    reason: str | None = None
    escalated: bool = False
    method_ok: bool | None = None
    flags: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    alignments: list[Alignment] = field(default_factory=list)
    packet_sha256: str | None = None
    b4_context: Literal["full", "truncated"] | None = None
    b4_rows_shown: int | None = None
    b4_rows_total: int | None = None
    calls: list[ModelCall] = field(default_factory=list)
    det_ms: int = 0
    esc_prep_ms: int = 0
    cost_billed_usd: float = 0.0
    cost_conservative_usd: float = 0.0
    latency_ms: int = 0
    spend_usd: float = 0.0
    config_hash: str = ""


@dataclass
class LabelledItem:
    """Dataset or derived benchmark item with ground truth and reference label."""

    item_id: str
    group_id: str
    split: Literal["calibration", "test"]
    set: Literal["S1", "S2", "S3", "S3m", "S4", "S5"]
    family: str
    slices: list[str]
    trace: Trace
    gt: GroundTruth
    label: Literal["correct", "incorrect", "ambiguous", "gt_wrong"]
    pair_of: str | None = None
    method_ok_label: bool | None = None
    snapshot: str = "minidev"


@dataclass
class RunIdentity:
    """Immutable run metadata written once to run_dir/run.json."""

    config_hash: str
    allow_remote: bool
    items_sha256: str
    manifest_sha256: str
    code_commit: str
    code_dirty: bool
    cache_only: bool


@dataclass
class Record:
    """Single terminal verdict entry written to run_dir/records.jsonl."""

    item_id: str
    method: RecordedMethodName
    verdict: Verdict


@dataclass
class Reservation:
    """In-flight spend reservation persisted prior to a billable attempt."""

    stage: str
    amount_usd: float


@dataclass
class SpendCheckpoint:
    """Experiment-wide spend tracker persisted atomically to spend.json."""

    cap_usd: float = 150.0
    committed_usd: float = 0.0
    by_stage: dict[str, float] = field(default_factory=dict)
    reserved: Reservation | None = None
    underestimates: int = 0


@dataclass
class ThresholdPair:
    """Abstain-band thresholds (tau_hi, tau_lo) for Jev probabilities."""

    tau_hi: float = 0.85
    tau_lo: float = 0.15


@dataclass
class ModelPricing:
    """Token prices in USD per 1M tokens."""

    input_per_m: float
    output_per_m: float


DEFAULT_JEV_MODEL = "jev-1.14.0"
DEFAULT_GENERATIVE_MODEL = "gemini-4-argon"
DEFAULT_AGENT_MODEL = "gemini-3.8-flash"

DEFAULT_PRICING: dict[str, ModelPricing] = {
    "jev-1.14.0": ModelPricing(input_per_m=0.042, output_per_m=0.0),
    "gemini-4-argon": ModelPricing(input_per_m=3.0, output_per_m=15.0),
    "gemini-3.8-flash": ModelPricing(input_per_m=0.75, output_per_m=3.75),
}


def default_tau_map() -> dict[str, dict[str, ThresholdPair]]:
    """Return default (tau_hi=0.85, tau_lo=0.15) thresholds per method and mode."""
    return {
        "H+E": {"sql_only": ThresholdPair(), "full_trace": ThresholdPair()},
        "H": {"sql_only": ThresholdPair(), "full_trace": ThresholdPair()},
        "B6": {"sql_only": ThresholdPair(), "full_trace": ThresholdPair()},
        "B6+E": {"sql_only": ThresholdPair(), "full_trace": ThresholdPair()},
    }


@dataclass
class EvalConfig:
    """Configuration controlling execution limits, model IDs, thresholds, and safety modes."""

    jev_model: str = DEFAULT_JEV_MODEL
    generative_model: str = DEFAULT_GENERATIVE_MODEL
    agent_model: str = DEFAULT_AGENT_MODEL
    escalation: bool = True
    enable_d5: bool = False
    local_only: bool = False
    cache_only: bool = False
    strict_manifest: bool = True

    # Sandbox & input limits (in config_hash)
    sql_timeout_s: float = 10.0
    fetch_cap_rows: int = 200_000
    fetch_cap_bytes: int = 16 * 1024 * 1024
    max_cell_bytes: int = 1024 * 1024
    max_columns: int = 64
    max_sql_bytes: int = 32 * 1024
    max_steps: int = 8
    max_answer_chars: int = 20_000
    sqlite_heap_limit_bytes: int = 256 * 1024 * 1024

    # Packet & context tiers (in config_hash)
    jev_max_cells: int = 400
    jev_max_tokens: int = 6_000
    jev_max_questions: int = 4
    escalation_max_cells: int = 1_200
    escalation_max_tokens: int = 16_000
    b4_max_tokens: int = 100_000

    # Retry & timeout policy (in config_hash)
    jev_max_attempts: int = 3
    jev_timeout_s: float = 10.0
    jev_backoff_s: tuple[float, ...] = (1.0, 2.0)
    generative_max_attempts: int = 2
    generative_timeout_s: float = 60.0
    max_retry_after_s: float = 30.0
    max_generative_output_tokens: int = 1024

    # Thresholds & pricing
    tau: dict[str, dict[str, ThresholdPair]] = field(default_factory=default_tau_map)
    pricing: dict[str, ModelPricing] = field(
        default_factory=lambda: dict(DEFAULT_PRICING)
    )

    # Spend cap is intentionally excluded from config_hash so caps can be adjusted on resume.
    spend_cap_usd: float = 150.0

    def get_pricing(self, model: str) -> ModelPricing:
        """Return ModelPricing for *model*, falling back to generative illustration rates."""
        if model in self.pricing:
            return self.pricing[model]
        if model.startswith("jev"):
            return self.pricing[DEFAULT_JEV_MODEL]
        if "flash" in model:
            return self.pricing[DEFAULT_AGENT_MODEL]
        return self.pricing[DEFAULT_GENERATIVE_MODEL]

    def config_hash(self, prompt_hash: str = "") -> str:
        """Compute deterministic SHA-256 over thresholds, prompts, models, and limits (excluding spend_cap_usd)."""
        payload = {
            "jev_model": self.jev_model,
            "generative_model": self.generative_model,
            "agent_model": self.agent_model,
            "escalation": self.escalation,
            "enable_d5": self.enable_d5,
            "sql_timeout_s": self.sql_timeout_s,
            "fetch_cap_rows": self.fetch_cap_rows,
            "fetch_cap_bytes": self.fetch_cap_bytes,
            "max_cell_bytes": self.max_cell_bytes,
            "max_columns": self.max_columns,
            "max_sql_bytes": self.max_sql_bytes,
            "max_steps": self.max_steps,
            "max_answer_chars": self.max_answer_chars,
            "sqlite_heap_limit_bytes": self.sqlite_heap_limit_bytes,
            "jev_max_cells": self.jev_max_cells,
            "jev_max_tokens": self.jev_max_tokens,
            "jev_max_questions": self.jev_max_questions,
            "escalation_max_cells": self.escalation_max_cells,
            "escalation_max_tokens": self.escalation_max_tokens,
            "b4_max_tokens": self.b4_max_tokens,
            "jev_max_attempts": self.jev_max_attempts,
            "jev_timeout_s": self.jev_timeout_s,
            "jev_backoff_s": list(self.jev_backoff_s),
            "generative_max_attempts": self.generative_max_attempts,
            "generative_timeout_s": self.generative_timeout_s,
            "max_retry_after_s": self.max_retry_after_s,
            "max_generative_output_tokens": self.max_generative_output_tokens,
            "tau": {
                m: {
                    mode: {"tau_hi": pair.tau_hi, "tau_lo": pair.tau_lo}
                    for mode, pair in modes.items()
                }
                for m, modes in sorted(self.tau.items())
            },
            "prompt_hash": prompt_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
