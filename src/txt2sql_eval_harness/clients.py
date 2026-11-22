"""Model clients, validation, retry/cost tracking, and strict egress-gated call_model."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Literal, Protocol

from txt2sql_eval_harness.cache import ResponseCache
from txt2sql_eval_harness.models import (
    Attempt,
    CallMeasure,
    EvalConfig,
    ModelCall,
)

BillingKind = Literal["reported", "estimated", "ambiguous", "none"]
StageName = Literal["jev", "escalation", "b4", "b6", "ibm_judge", "agent", "prompt_check"]
from txt2sql_eval_harness.renderer import count_tokens
from txt2sql_eval_harness.rubrics import RUBRICS
from txt2sql_eval_harness.spend import SpendLedger

TRANSIENT_RESULTS: frozenset[str] = frozenset(
    {"timeout", "connection_lost", "connect_refused", "http_429", "http_5xx"}
)
VALID_GENERATIVE_LABELS: frozenset[str] = frozenset({"CORRECT", "INCORRECT", "AMBIGUOUS"})
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

_IBM_VERDICT_RE = re.compile(
    r"^\s*(?:[*_#>\-\s]*)(?:(?:verdict|answer)\s*[:\-]\s*)?(yes|no|maybe)\b",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class RawModelResponse:
    """Raw HTTP or SDK response container returned by a `ModelTransport` before validation."""

    http_ok: bool
    status_code: int | None = 200
    transport_error: Literal["timeout", "connection_lost", "connect_refused"] | None = None
    payload: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reported_cost_usd: float | None = None
    retry_after_s: float | None = None


class ModelTransport(Protocol):
    """Transport protocol implemented by live HTTP/SDK clients and deterministic test fakes."""

    def send(self, request: dict[str, Any], *, timeout: float) -> RawModelResponse:
        """Send a serialized request payload to the remote backend and return a RawModelResponse."""
        ...


@dataclass
class CallContext:
    """Runtime execution context governing remote egress, response caching, spend caps, and clocks."""

    cfg: EvalConfig
    allow_remote: bool
    cache_only: bool = False
    cache: ResponseCache | None = None
    ledger: SpendLedger | None = None
    clients: dict[str, ModelTransport] = field(default_factory=dict)
    sleep_fn: Callable[[float], Any] = time.sleep
    clock_fn: Callable[[], float] = time.monotonic


def canonical_request_sha256(backend: str, model: str, request: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 cache key over backend, model, and canonical JSON request."""
    payload = {
        "backend": backend,
        "model": model,
        "request": request,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_prob(val: Any) -> bool:
    """Return whether `val` is a finite non-boolean numeric probability in [0.0, 1.0]."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(float(val))
        and 0.0 <= float(val) <= 1.0
    )


def validate_jev_response(
    payload: Any,
    requested_questions: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """All-or-nothing validator for TypeSafe Jev System One response."""
    if not isinstance(payload, dict):
        return None, "jev_payload_not_dict"

    raw_answers: Any = payload
    for container_key in ("answers", "results", "questions"):
        if container_key in payload and isinstance(payload[container_key], dict):
            raw_answers = payload[container_key]
            break

    if not isinstance(raw_answers, dict):
        return None, "jev_missing_answers_map"

    normalized: dict[str, Any] = {}
    for qname in requested_questions:
        if qname not in raw_answers:
            return None, f"jev_missing_question:{qname}"
        rubric = RUBRICS.get(qname)
        if rubric is None:
            return None, f"jev_unknown_rubric:{qname}"

        entry = raw_answers[qname]
        if rubric.qtype == "noul":
            p_val: Any = entry
            if isinstance(entry, dict):
                for key in ("probability", "p_true", "noul", "score", "value"):
                    if key in entry:
                        p_val = entry[key]
                        break
            if not _is_prob(p_val):
                return None, f"jev_invalid_noul:{qname}"
            normalized[qname] = float(p_val)
        else:
            probs_raw: Any = entry
            if isinstance(entry, dict) and "probabilities" in entry:
                probs_raw = entry["probabilities"]
            if not isinstance(probs_raw, dict):
                return None, f"jev_invalid_choice:{qname}"
            expected_opts = (
                set(rubric.criteria.keys())
                if isinstance(rubric.criteria, dict)
                else set(rubric.criteria or [])
            )
            if set(probs_raw.keys()) != expected_opts:
                return None, f"jev_choice_options_mismatch:{qname}"
            probs_norm: dict[str, float] = {}
            total_p = 0.0
            for opt, p in probs_raw.items():
                if not _is_prob(p):
                    return None, f"jev_invalid_choice_prob:{qname}:{opt}"
                fp = float(p)
                probs_norm[opt] = fp
                total_p += fp
            if not (0.99 <= total_p <= 1.01):
                return None, f"jev_choice_prob_sum:{qname}:{total_p}"
            normalized[qname] = probs_norm

    return normalized, None


def _strip_json_fence(text: str) -> str:
    """Remove an enclosing Markdown triple-backtick fence if present around JSON text."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return s


def validate_generative_label_response(
    payload: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate JSON response containing exact case-sensitive CORRECT/INCORRECT/AMBIGUOUS label."""
    obj: Any
    if isinstance(payload, dict):
        obj = payload
    elif isinstance(payload, str):
        try:
            obj = json.loads(_strip_json_fence(payload))
        except Exception as exc:
            return None, f"invalid_json:{exc}"
    else:
        return None, "generative_payload_type"

    if not isinstance(obj, dict):
        return None, "generative_not_object"
    label = obj.get("label")
    if not isinstance(label, str) or label not in VALID_GENERATIVE_LABELS:
        return None, f"invalid_label:{label!r}"
    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        return None, "invalid_rationale"
    return {"label": label, "rationale": rationale}, None


def validate_ibm_judge_response(
    payload: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate text response for IBM judge regex verdict (Yes|No|Maybe)."""
    if not isinstance(payload, str) or not payload.strip():
        return None, "empty_ibm_response"
    m = _IBM_VERDICT_RE.search(payload)
    if m is None:
        return None, "missing_ibm_verdict"
    verdict = m.group(1).capitalize()
    return {"verdict": verdict, "raw_text": payload}, None


def validate_agent_response(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate agent turn output: expects SQL tool call or FINAL_ANSWER string/dict."""
    if isinstance(payload, dict):
        if "text" in payload and isinstance(payload["text"], str):
            return {"text": payload["text"]}, None
        if "sql" in payload or "final_answer" in payload:
            return payload, None
    elif isinstance(payload, str) and payload.strip():
        return {"text": payload}, None
    return None, "invalid_agent_response"


class JevHttpClient:
    """HTTP transport for TypeSafe Jev decision API (imports httpx lazily inside method)."""

    def __init__(self, api_key: str, model: str, endpoint: str = JEV_ENDPOINT) -> None:
        """Configure the Jev HTTP client with an API key, model name, and endpoint URL."""
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def send(self, request: dict[str, Any], *, timeout: float) -> RawModelResponse:
        """POST a single Jev evaluation request and normalize transport or HTTP errors."""
        import httpx

        body = {
            "model": self.model,
            "state": request["state"],
            "questions": request["questions"],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(self.endpoint, json=body, headers=headers)
        except httpx.TimeoutException:
            return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")
        except httpx.ConnectError as exc:
            msg = str(exc).lower()
            err_kind: Literal["connection_lost", "connect_refused"] = (
                "connect_refused" if "refused" in msg else "connection_lost"
            )
            return RawModelResponse(http_ok=False, status_code=None, transport_error=err_kind)
        except httpx.HTTPError:
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connection_lost"
            )

        retry_after_s: float | None = None
        if "Retry-After" in resp.headers:
            try:
                retry_after_s = float(resp.headers["Retry-After"])
            except ValueError:
                retry_after_s = None

        if resp.status_code != 200:
            return RawModelResponse(
                http_ok=False,
                status_code=resp.status_code,
                retry_after_s=retry_after_s,
            )

        try:
            data = resp.json()
        except Exception:
            return RawModelResponse(http_ok=True, status_code=200, payload=resp.text)

        usage = data.get("usage") if isinstance(data, dict) else None
        in_tok = (
            int(usage["input_tokens"])
            if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int)
            else None
        )
        rep_cost = (
            float(usage["cost_usd"])
            if isinstance(usage, dict)
            and isinstance(usage.get("cost_usd"), (int, float))
            and not isinstance(usage.get("cost_usd"), bool)
            else None
        )
        return RawModelResponse(
            http_ok=True,
            status_code=200,
            payload=data,
            input_tokens=in_tok,
            output_tokens=0,
            reported_cost_usd=rep_cost,
        )


class GeminiGenerativeClient:
    """Generative transport using google-genai SDK (imports google.genai & reads key lazily)."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the Gemini client using the configured API key and target model."""
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def send(self, request: dict[str, Any], *, timeout: float) -> RawModelResponse:
        """Send a Gemini generative request and convert transport or API failures into RawModelResponse."""
        from google.genai import errors as genai_errors
        from google.genai import types

        system_instruction = request.get("instructions") or request.get("system")
        if "prompt" in request:
            contents: Any = request["prompt"]
        elif "messages" in request:
            contents = request["messages"]
        else:
            contents = request.get("state", "")

        temperature = float(request.get("temperature", 0.0))
        max_out = int(request.get("max_output_tokens", 512))

        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_out,
            "http_options": types.HttpOptions(timeout=int(max(1, round(timeout * 1000)))),
        }
        if system_instruction:
            cfg_kwargs["system_instruction"] = str(system_instruction)

        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        except TimeoutError:
            return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")
        except ConnectionRefusedError:
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connect_refused"
            )
        except ConnectionError:
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connection_lost"
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            status = int(code) if isinstance(code, int) else 500
            return RawModelResponse(http_ok=False, status_code=status)
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
                return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connection_lost"
            )

        text = getattr(resp, "text", None) or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) if usage is not None else None
        out_tok = getattr(usage, "candidates_token_count", None) if usage is not None else None
        thoughts_tok = getattr(usage, "thoughts_token_count", None) if usage is not None else None
        total_out = None
        if out_tok is not None or thoughts_tok is not None:
            total_out = int(out_tok or 0) + int(thoughts_tok or 0)

        return RawModelResponse(
            http_ok=True,
            status_code=200,
            payload=text,
            input_tokens=int(in_tok) if in_tok is not None else None,
            output_tokens=total_out,
        )


def _get_api_key(var_name: str) -> str | None:
    """Resolve an API key from process environment first, falling back to local `.env` files."""
    val = os.environ.get(var_name)
    if val:
        return val
    candidates = (
        Path(".env"),
        Path("src/.env"),
        Path(__file__).resolve().parent.parent / ".env",
    )
    line_re = re.compile(rf"^\s*{re.escape(var_name)}\s*=\s*(.+?)\s*$")
    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            text = env_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = line_re.match(raw_line)
            if not m:
                continue
            raw_val = m.group(1).strip()
            if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in ("'", '"'):
                raw_val = raw_val[1:-1]
            if raw_val:
                return raw_val
    return None


_CLIENT_LOCK = threading.Lock()


class GeminiDecisionClient:
    """Decision transport using google-genai Decisions API (gemini-decision-1)."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the Gemini Decisions client using GEMINI_API_KEY and v1beta API."""
        from google import genai

        self.model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )

    @staticmethod
    def _normalize_response_answers(resp: Any) -> dict[str, Any]:
        """Convert CreateDecisionResponse.answers into a dict compatible with validate_jev_response."""
        raw_answers = getattr(resp, "answers", None) or {}
        out: dict[str, Any] = {}
        for qname, ans in raw_answers.items():
            if hasattr(ans, "noul") and getattr(ans, "noul") is not None:
                out[str(qname)] = {"noul": float(ans.noul)}
            elif hasattr(ans, "probabilities") and getattr(ans, "probabilities") is not None:
                out[str(qname)] = {
                    "probabilities": {str(k): float(v) for k, v in ans.probabilities.items()}
                }
        return out

    def send(self, request: dict[str, Any], *, timeout: float) -> RawModelResponse:
        """Send a Gemini Decisions API request and normalize the response."""
        from google.genai import errors as genai_errors

        try:
            resp = self._client.decisions.create(
                model=self.model,
                input=request["state"],
                questions=request["questions"],
            )
        except TimeoutError:
            return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")
        except ConnectionRefusedError:
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connect_refused"
            )
        except ConnectionError:
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connection_lost"
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            status = int(code) if isinstance(code, int) else 500
            return RawModelResponse(http_ok=False, status_code=status)
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
                return RawModelResponse(http_ok=False, status_code=None, transport_error="timeout")
            return RawModelResponse(
                http_ok=False, status_code=None, transport_error="connection_lost"
            )

        answers = self._normalize_response_answers(resp)
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "total_input_tokens", None) if usage is not None else None
        return RawModelResponse(
            http_ok=True,
            status_code=200,
            payload={"answers": answers},
            input_tokens=int(in_tok) if in_tok is not None else None,
            output_tokens=0,
        )


def client_for(backend: str, model: str, ctx: CallContext) -> ModelTransport:
    """Lazy factory — ONLY location permitted to read API-key env vars or construct SDK clients."""
    with _CLIENT_LOCK:
        if backend in ctx.clients:
            return ctx.clients[backend]
        if backend == "jev":
            if model.startswith("gemini-decision"):
                api_key = _get_api_key("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "GEMINI_API_KEY is required for remote Gemini Decision calls"
                    )
                dec_client: ModelTransport = GeminiDecisionClient(api_key=api_key, model=model)
                ctx.clients[backend] = dec_client
                return dec_client
            api_key = _get_api_key("TYPESAFE_API_KEY")
            if not api_key:
                raise RuntimeError("TYPESAFE_API_KEY is required for remote Jev calls")
            client: ModelTransport = JevHttpClient(api_key=api_key, model=model)
            ctx.clients[backend] = client
            return client
        if backend in ("generative", "ibm_judge"):
            api_key = _get_api_key("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required for remote generative calls")
            gen_client: ModelTransport = GeminiGenerativeClient(api_key=api_key, model=model)
            ctx.clients[backend] = gen_client
            return gen_client
        raise ValueError(f"Unknown backend: {backend!r}")


@dataclass(frozen=True)
class _RetryPolicy:
    """Per-stage retry count, timeout budget, backoff schedule, and invalid-payload limit."""

    max_attempts: int
    timeout_s: float
    backoffs_s: tuple[float, ...]
    max_invalid: int


def _policy_for_stage(stage: StageName, cfg: EvalConfig) -> _RetryPolicy:
    """Return the retry and timeout policy for the specified judge stage from EvalConfig."""
    if stage == "jev":
        return _RetryPolicy(
            max_attempts=cfg.jev_max_attempts,
            timeout_s=cfg.jev_timeout_s,
            backoffs_s=tuple(cfg.jev_backoff_s),
            max_invalid=2,
        )
    return _RetryPolicy(
        max_attempts=cfg.generative_max_attempts,
        timeout_s=cfg.generative_timeout_s,
        backoffs_s=(),
        max_invalid=2,
    )


def _upper_bound_usd(cfg: EvalConfig, model: str, request: dict[str, Any], stage: StageName) -> tuple[float, int]:
    """Compute conservative worst-case USD cost and estimated input tokens before sending a request."""
    req_str = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    local_in = count_tokens(req_str)
    pricing = cfg.get_pricing(model)
    max_out = 0 if stage == "jev" else int(request.get("max_output_tokens", 512))
    upper = (
        (1.2 * local_in * pricing.input_per_m) + (max_out * pricing.output_per_m)
    ) / 1_000_000.0
    return upper, local_in


def _classify_http_status(status: int | None) -> str:
    """Classify a non-200 HTTP status code into retryable vs. terminal error categories."""
    if status == 429:
        return "http_429"
    if status is not None and 500 <= status <= 599:
        return "http_5xx"
    return "http_4xx"


def _compute_wait_s(
    policy: _RetryPolicy,
    attempt_num: int,
    prev: RawModelResponse | None,
    max_retry_after_s: float = 30.0,
) -> float:
    """Compute pre-attempt sleep duration in seconds honoring capped `Retry-After` headers."""
    if attempt_num <= 1:
        return 0.0
    if prev is not None and prev.retry_after_s is not None and prev.retry_after_s >= 0:
        return min(float(prev.retry_after_s), max_retry_after_s)
    idx = attempt_num - 2
    if idx < len(policy.backoffs_s):
        return policy.backoffs_s[idx]
    return 0.0


def _validate_stage(
    stage: StageName,
    payload: Any,
    question_names: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Dispatch payload validation to the strict parser matching `stage`."""
    if stage == "jev":
        return validate_jev_response(payload, question_names)
    if stage == "ibm_judge":
        return validate_ibm_judge_response(payload)
    if stage == "agent":
        return validate_agent_response(payload)
    return validate_generative_label_response(payload)


def _build_attempt_cost(
    cfg: EvalConfig,
    model: str,
    stage: StageName,
    resp: RawModelResponse,
    result_kind: str,
    local_in: int,
    upper_usd: float,
) -> tuple[int, int | None, BillingKind, float]:
    """Compute tokens, billing classification, and USD cost for one completed attempt."""
    pricing = cfg.get_pricing(model)
    in_tok = resp.input_tokens if resp.input_tokens is not None else local_in
    out_tok = resp.output_tokens

    if result_kind in ("timeout", "connection_lost"):
        return in_tok, out_tok, "ambiguous", upper_usd
    if result_kind in ("connect_refused", "http_4xx", "http_429", "http_5xx"):
        return in_tok, out_tok, "none", 0.0

    # HTTP 200 (ok or invalid)
    if resp.reported_cost_usd is not None:
        return in_tok, out_tok, "reported", float(resp.reported_cost_usd)

    eff_out = (
        0
        if stage == "jev"
        else (
            out_tok
            if out_tok is not None
            else (
                count_tokens(resp.payload)
                if isinstance(resp.payload, str)
                else count_tokens(json.dumps(resp.payload, ensure_ascii=True))
                if resp.payload is not None
                else 0
            )
        )
    )
    est = ((in_tok * pricing.input_per_m) + (eff_out * pricing.output_per_m)) / 1_000_000.0
    return in_tok, eff_out, "estimated", est


def _summarize_measure(attempts: list[Attempt]) -> CallMeasure:
    """Aggregate total wall time, billed cost, and ambiguous cost across transport attempts."""
    wall_ms = sum(a.wait_ms + a.elapsed_ms for a in attempts)
    billed = sum(a.cost_usd for a in attempts if a.billing in ("reported", "estimated"))
    ambig = sum(a.cost_usd for a in attempts if a.billing == "ambiguous")
    return CallMeasure(
        attempts=attempts,
        wall_ms=wall_ms,
        billed_usd=billed,
        ambiguous_usd=ambig,
    )


def call_model(
    stage: StageName,
    backend: str,
    model: str,
    request: dict[str, Any],
    question_names: list[str],
    ctx: CallContext,
) -> ModelCall:
    """Single egress-gated entrypoint for all model calls across the harness.

    Strict sequence:
    1. canonical request_sha256
    2. egress gate (`not ctx.allow_remote` -> `remote_disabled` BEFORE cache, client, keys, ledger)
    3. cache lookup (`from_cache=True`, cold measurement preserved)
    4. `ctx.cache_only` -> `cache_miss`
    5. lazy `client_for` -> retry loop with pre-send reserve & post-send settle
    6. write-once `cache.put_once` strictly on validated `"ok"`
    """
    req_sha = canonical_request_sha256(backend, model, request)
    zero_measure = CallMeasure(attempts=[], wall_ms=0, billed_usd=0.0, ambiguous_usd=0.0)

    if not ctx.allow_remote:
        return ModelCall(
            stage=stage,
            model=model,
            request_sha256=req_sha,
            questions=list(question_names),
            answers=None,
            outcome="remote_disabled",
            invalid_reason=None,
            from_cache=False,
            cold=zero_measure,
        )

    if ctx.cache is not None:
        cached = ctx.cache.get(req_sha)
        if cached is not None:
            return ModelCall(
                stage=stage,
                model=model,
                request_sha256=req_sha,
                questions=list(question_names),
                answers=cached.answers,
                outcome="ok",
                invalid_reason=None,
                from_cache=True,
                cold=cached.cold,
            )

    if ctx.cache_only:
        return ModelCall(
            stage=stage,
            model=model,
            request_sha256=req_sha,
            questions=list(question_names),
            answers=None,
            outcome="cache_miss",
            invalid_reason=None,
            from_cache=False,
            cold=zero_measure,
        )

    client = client_for(backend, model, ctx)
    policy = _policy_for_stage(stage, ctx.cfg)
    upper_usd, local_in = _upper_bound_usd(ctx.cfg, model, request, stage)

    attempts: list[Attempt] = []
    invalid_count = 0
    last_invalid_reason: str | None = None
    last_raw: RawModelResponse | None = None

    for n in range(1, policy.max_attempts + 1):
        wait_s = _compute_wait_s(policy, n, last_raw, ctx.cfg.max_retry_after_s)
        if wait_s > 0.0:
            ctx.sleep_fn(wait_s)
        wait_ms = int(round(wait_s * 1000))

        if ctx.ledger is not None:
            if not ctx.ledger.reserve(stage, upper_usd):
                return ModelCall(
                    stage=stage,
                    model=model,
                    request_sha256=req_sha,
                    questions=list(question_names),
                    answers=None,
                    outcome="spend_cap",
                    invalid_reason=None,
                    from_cache=False,
                    cold=_summarize_measure(attempts),
                )

        t0 = ctx.clock_fn()
        try:
            raw = client.send(request, timeout=policy.timeout_s)
        except Exception:
            raw = RawModelResponse(
                http_ok=False, status_code=None, transport_error="connection_lost"
            )
        elapsed_ms = max(0, int(round((ctx.clock_fn() - t0) * 1000)))
        last_raw = raw

        validated: dict[str, Any] | None = None
        if raw.transport_error is not None:
            result_kind = raw.transport_error
        elif not raw.http_ok:
            result_kind = _classify_http_status(raw.status_code)
        else:
            validated, invalid_reason = _validate_stage(stage, raw.payload, question_names)
            if validated is not None:
                result_kind = "ok"
            else:
                result_kind = "invalid"
                invalid_count += 1
                last_invalid_reason = invalid_reason

        in_tok, out_tok, billing, cost_usd = _build_attempt_cost(
            ctx.cfg, model, stage, raw, result_kind, local_in, upper_usd
        )

        if ctx.ledger is not None:
            ctx.ledger.settle(cost_usd, stage=stage, reserved_usd=upper_usd)

        attempts.append(
            Attempt(
                n=n,
                wait_ms=wait_ms,
                elapsed_ms=elapsed_ms,
                result=result_kind,  # type: ignore[arg-type]
                input_tokens=in_tok,
                output_tokens=out_tok,
                billing=billing,
                cost_usd=cost_usd,
            )
        )

        if result_kind == "ok" and validated is not None:
            cold = _summarize_measure(attempts)
            if ctx.cache is not None:
                ctx.cache.put_once(req_sha, stage, model, validated, cold)
            return ModelCall(
                stage=stage,
                model=model,
                request_sha256=req_sha,
                questions=list(question_names),
                answers=validated,
                outcome="ok",
                invalid_reason=None,
                from_cache=False,
                cold=cold,
            )

        if result_kind == "invalid" and invalid_count >= policy.max_invalid:
            return ModelCall(
                stage=stage,
                model=model,
                request_sha256=req_sha,
                questions=list(question_names),
                answers=None,
                outcome="invalid_response",
                invalid_reason=last_invalid_reason,
                from_cache=False,
                cold=_summarize_measure(attempts),
            )

        if result_kind not in TRANSIENT_RESULTS and result_kind != "invalid":
            return ModelCall(
                stage=stage,
                model=model,
                request_sha256=req_sha,
                questions=list(question_names),
                answers=None,
                outcome="backend_error",
                invalid_reason=None,
                from_cache=False,
                cold=_summarize_measure(attempts),
            )

    final_outcome: Literal["invalid_response", "backend_error"] = (
        "invalid_response" if attempts and attempts[-1].result == "invalid" else "backend_error"
    )
    return ModelCall(
        stage=stage,
        model=model,
        request_sha256=req_sha,
        questions=list(question_names),
        answers=None,
        outcome=final_outcome,
        invalid_reason=last_invalid_reason if final_outcome == "invalid_response" else None,
        from_cache=False,
        cold=_summarize_measure(attempts),
    )
