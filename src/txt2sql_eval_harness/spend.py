"""Durable spend checkpoint and non-blocking process lock per experiment root."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
from pathlib import Path
import threading
from typing import Any

from txt2sql_eval_harness.models import Reservation, SpendCheckpoint


class ExperimentBusyError(RuntimeError):
    """Raised when <root>/spend.lock is already held by another live process."""


class SpendCheckpointCorruptError(ValueError):
    """Raised when <root>/spend.json cannot be parsed or fails schema validation."""


def _is_num(val: Any) -> bool:
    """Return True iff *val* is a finite non-boolean numeric value."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(float(val))
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via temp file + fsync + os.replace + directory fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    # Fsync parent directory so the rename is durable across sudden power loss.
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _decode_checkpoint(raw_text: str) -> SpendCheckpoint:
    """Parse and validate a spend.json payload into a SpendCheckpoint."""
    try:
        d = json.loads(raw_text)
    except Exception as exc:
        raise SpendCheckpointCorruptError("spend_checkpoint_corrupt") from exc
    if not isinstance(d, dict):
        raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")

    required = {"cap_usd", "committed_usd", "by_stage", "reserved", "underestimates"}
    if not required.issubset(d.keys()):
        raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")

    if not (_is_num(d["cap_usd"]) and _is_num(d["committed_usd"])):
        raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")
    under = d["underestimates"]
    if not isinstance(under, int) or isinstance(under, bool) or under < 0:
        raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")
    by_stage_raw = d["by_stage"]
    if not isinstance(by_stage_raw, dict):
        raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")
    by_stage: dict[str, float] = {}
    for k, v in by_stage_raw.items():
        if not isinstance(k, str) or not _is_num(v):
            raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")
        by_stage[k] = float(v)

    res_raw = d["reserved"]
    reserved: Reservation | None = None
    if res_raw is not None:
        if (
            not isinstance(res_raw, dict)
            or not isinstance(res_raw.get("stage"), str)
            or not _is_num(res_raw.get("amount_usd"))
        ):
            raise SpendCheckpointCorruptError("spend_checkpoint_corrupt")
        reserved = Reservation(
            stage=res_raw["stage"],
            amount_usd=float(res_raw["amount_usd"]),
        )

    return SpendCheckpoint(
        cap_usd=float(d["cap_usd"]),
        committed_usd=float(d["committed_usd"]),
        by_stage=by_stage,
        reserved=reserved,
        underestimates=under,
    )


def _encode_checkpoint(cp: SpendCheckpoint) -> dict[str, Any]:
    """Serialize a SpendCheckpoint into a JSON-ready dictionary."""
    return {
        "cap_usd": cp.cap_usd,
        "committed_usd": cp.committed_usd,
        "by_stage": cp.by_stage,
        "reserved": (
            {"stage": cp.reserved.stage, "amount_usd": cp.reserved.amount_usd}
            if cp.reserved is not None
            else None
        ),
        "underestimates": cp.underestimates,
    }


class SpendLedger:
    """Flocked, fsynced spend tracker supporting crash-safe pre-send reservation."""

    def __init__(self, root: Path, *, cap_usd: float = 150.0) -> None:
        """Acquire exclusive lock on <root>/spend.lock and load or initialize spend.json."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / "spend.lock"
        self.json_path = self.root / "spend.json"
        self._lock = threading.Lock()
        self._active_reservations: list[Reservation] = []
        self._lock_fd: int | None = None
        self._acquire_lock()
        try:
            self._checkpoint = self._open_or_init(cap_usd)
        except Exception:
            self.close()
            raise

    def _acquire_lock(self) -> None:
        """Acquire a non-blocking exclusive flock on spend.lock or raise ExperimentBusyError."""
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise ExperimentBusyError("experiment_busy") from exc
            raise
        self._lock_fd = fd

    def _open_or_init(self, cap_usd: float) -> SpendCheckpoint:
        """Open spend.json (folding any crash-orphaned reservation) or initialize a fresh ledger."""
        if self.json_path.exists():
            raw = self.json_path.read_text(encoding="utf-8")
            cp = _decode_checkpoint(raw)
            changed = False
            if cp.cap_usd != float(cap_usd):
                cp.cap_usd = float(cap_usd)
                changed = True
            if cp.reserved is not None:
                # Fold any interrupted reservation into orphans on restart so spend cannot be forgotten.
                orphan_amt = cp.reserved.amount_usd
                cp.committed_usd += orphan_amt
                cp.by_stage["orphan"] = cp.by_stage.get("orphan", 0.0) + orphan_amt
                cp.reserved = None
                changed = True
            if changed:
                _atomic_write_json(self.json_path, _encode_checkpoint(cp))
            return cp
        cp = SpendCheckpoint(cap_usd=float(cap_usd))
        _atomic_write_json(self.json_path, _encode_checkpoint(cp))
        return cp

    @property
    def checkpoint(self) -> SpendCheckpoint:
        """Return the live in-memory SpendCheckpoint snapshot."""
        return self._checkpoint

    def reserve(self, stage: str, upper_usd: float) -> bool:
        """Reserve upper_usd before sending; return False without writing if over cap."""
        if upper_usd < 0 or not math.isfinite(upper_usd):
            raise ValueError(f"Invalid upper bound: {upper_usd!r}")
        with self._lock:
            active_sum = sum(r.amount_usd for r in self._active_reservations)
            if self._checkpoint.committed_usd + active_sum + upper_usd > self._checkpoint.cap_usd + 1e-12:
                return False
            res = Reservation(stage=stage, amount_usd=float(upper_usd))
            self._active_reservations.append(res)
            self._checkpoint.reserved = Reservation(
                stage=stage,
                amount_usd=active_sum + float(upper_usd),
            )
            _atomic_write_json(self.json_path, _encode_checkpoint(self._checkpoint))
            return True

    def settle(
        self,
        actual_cost_usd: float,
        *,
        stage: str | None = None,
        reserved_usd: float | None = None,
    ) -> None:
        """Settle a matching active reservation atomically after an attempt completes."""
        with self._lock:
            res: Reservation | None = None
            if self._active_reservations:
                match_idx = 0
                for idx, cand in enumerate(self._active_reservations):
                    if stage is not None and cand.stage != stage:
                        continue
                    if reserved_usd is not None and abs(cand.amount_usd - float(reserved_usd)) > 1e-12:
                        continue
                    match_idx = idx
                    break
                res = self._active_reservations.pop(match_idx)
            elif self._checkpoint.reserved is not None:
                res = self._checkpoint.reserved
            else:
                raise RuntimeError("No active reservation to settle")

            cost = float(actual_cost_usd)
            self._checkpoint.committed_usd += cost
            self._checkpoint.by_stage[res.stage] = (
                self._checkpoint.by_stage.get(res.stage, 0.0) + cost
            )
            if cost > res.amount_usd + 1e-12:
                self._checkpoint.underestimates += 1
            if self._active_reservations:
                rem_sum = sum(r.amount_usd for r in self._active_reservations)
                self._checkpoint.reserved = Reservation(
                    stage=self._active_reservations[-1].stage,
                    amount_usd=rem_sum,
                )
            else:
                self._checkpoint.reserved = None
            _atomic_write_json(self.json_path, _encode_checkpoint(self._checkpoint))

    def close(self) -> None:
        """Release spend.lock and close its file descriptor."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def __enter__(self) -> SpendLedger:
        """Enter context manager returning self."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager and release spend.lock."""
        self.close()
