"""Text2SQL Decision-Model Judge evaluation harness."""

from txt2sql_eval_harness.models import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_GENERATIVE_MODEL,
    DEFAULT_JEV_MODEL,
    EvalConfig,
    GroundTruth,
    LabelledItem,
    ResultTable,
    Step,
    ThresholdPair,
    Trace,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_GENERATIVE_MODEL",
    "DEFAULT_JEV_MODEL",
    "EvalConfig",
    "GroundTruth",
    "LabelledItem",
    "ResultTable",
    "Step",
    "ThresholdPair",
    "Trace",
    "Verdict",
    "__version__",
]
