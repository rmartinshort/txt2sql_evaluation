"""Dataset manifest, egress admission gate, deterministic splits, and BIRD/FLEX loaders."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Literal, Sequence

from txt2sql_eval_harness.models import EvalConfig, GroundTruth, LabelledItem, Trace, Step


class UnadmittedInputError(RuntimeError):
    """Raised when a model-visible input file is not admitted by data/manifest.json."""


BIRD_MINIDEV_REPO = "birdsql/bird_mini_dev"
BIRD_MINIDEV_REVISION = "5b256b5a9c17478f06ed6912612e9a14d17065c8"
BIRD_MINIDEV_FILE = "archive/bird_mini_dev_2025-07-04/data/mini_dev_sqlite-00000-of-00001.json"
BIRD_MINIDEV_SHA256 = "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2"

FLEX_COMMIT = "270abdba59306134adf0928b2ce92acb8fbd94eb"
DEFAULT_MINIDEV_DB_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG&export=download&confirm=t"
)


def norm_question(question: str) -> str:
    """Normalize question text for group_id hashing: NFC, casefold, collapse WS, strip ?.!. """
    text = unicodedata.normalize("NFC", question).casefold()
    text = " ".join(text.split())
    return text.rstrip("?.!")


def compute_group_id(db_id: str, question: str) -> str:
    """Compute 16-hex group_id = sha256(db_id + '\\x1f' + norm(question))[:16]."""
    payload = f"{db_id}\x1f{norm_question(question)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def assign_split(group_id: str, seed: str = "split-v1") -> Literal["calibration", "test"]:
    """Assign calibration/test split deterministically by seeded hash of group_id (50/50)."""
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    return "calibration" if (int(digest[:8], 16) % 2 == 0) else "test"


def assert_disjoint_splits(items: Sequence[LabelledItem]) -> None:
    """Abort if any group_id appears in both calibration and test splits."""
    seen: dict[str, str] = {}
    for item in items:
        prev = seen.get(item.group_id)
        if prev is not None and prev != item.split:
            raise ValueError(
                f"Split leakage detected: group_id {item.group_id!r} in both {prev} and {item.split}"
            )
        seen[item.group_id] = item.split


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of *path* in streaming 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load data/manifest.json or return an empty manifest structure if absent."""
    if not manifest_path.exists():
        return {"entries": {}, "databases": {}}
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("entries", {})
    data.setdefault("databases", {})
    return data


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Persist data/manifest.json deterministically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(manifest_path)


def _manifest_key(file_path: Path, base_dir: Path | None = None) -> str:
    """Return a canonical relative POSIX manifest key (or resolved path fallback) for *file_path*."""
    resolved = file_path.resolve()
    if base_dir is not None:
        try:
            return resolved.relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def record_manifest_entry(
    manifest_path: Path,
    file_path: Path,
    kind: Literal["pinned_download", "harness_derivative"],
    *,
    base_dir: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Compute SHA-256 of *file_path* and record it in *manifest_path*."""
    digest = sha256_file(file_path)
    manifest = load_manifest(manifest_path)
    key = _manifest_key(file_path, base_dir=base_dir or manifest_path.parent.parent)
    entry: dict[str, Any] = {
        "sha256": digest,
        "kind": kind,
        "resolved_path": file_path.resolve().as_posix(),
    }
    if meta:
        entry["meta"] = meta
    manifest["entries"][key] = entry
    save_manifest(manifest_path, manifest)
    return digest


def record_database_entry(
    manifest_path: Path,
    db_id: str,
    db_path: Path,
    snapshot: str = "minidev",
) -> dict[str, Any]:
    """Record database sha256, trigger count, and view count in manifest.json."""
    digest = sha256_file(db_path)
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        triggers = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
        ).fetchone()[0]
        views = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='view'"
        ).fetchone()[0]
    finally:
        conn.close()
    manifest = load_manifest(manifest_path)
    manifest["databases"][f"{snapshot}:{db_id}"] = {
        "db_id": db_id,
        "snapshot": snapshot,
        "sha256": digest,
        "triggers": int(triggers),
        "views": int(views),
    }
    manifest["entries"][_manifest_key(db_path, base_dir=manifest_path.parent.parent)] = {
        "sha256": digest,
        "kind": "pinned_download",
        "resolved_path": db_path.resolve().as_posix(),
    }
    save_manifest(manifest_path, manifest)
    return manifest["databases"][f"{snapshot}:{db_id}"]


def egress_decision(
    input_files: Sequence[Path],
    manifest_path: Path,
    *,
    local_only: bool = False,
    base_dir: Path | None = None,
) -> bool:
    """Enforce the manifest-based source boundary once at run start.

    A model-visible file is admitted iff *manifest_path* lists it with matching
    whole-file sha256 and kind in {"pinned_download", "harness_derivative"}.
    Never promotes based on item_id, path alone, db_id, or db_sha256.
    """
    if not manifest_path.exists():
        if not local_only:
            raise UnadmittedInputError(
                f"unadmitted_input: manifest not found at {manifest_path}"
            )
        return False

    manifest = load_manifest(manifest_path)
    entries = manifest.get("entries", {})
    root = base_dir or manifest_path.parent.parent

    # Build lookup by both relative key and resolved path
    by_resolved: dict[str, dict[str, Any]] = {}
    for k, entry in entries.items():
        if "resolved_path" in entry:
            by_resolved[entry["resolved_path"]] = entry
        p = (root / k).resolve()
        by_resolved[p.as_posix()] = entry

    all_admitted = True
    unadmitted_reasons: list[str] = []

    for fpath in input_files:
        if not fpath.exists():
            all_admitted = False
            unadmitted_reasons.append(f"missing file: {fpath}")
            continue
        resolved_str = fpath.resolve().as_posix()
        rel_key = _manifest_key(fpath, base_dir=root)
        entry = entries.get(rel_key) or by_resolved.get(resolved_str)
        if entry is None:
            all_admitted = False
            unadmitted_reasons.append(f"not in manifest: {fpath}")
            continue
        if entry.get("kind") not in ("pinned_download", "harness_derivative"):
            all_admitted = False
            unadmitted_reasons.append(f"disallowed kind {entry.get('kind')!r} for {fpath}")
            continue
        actual_sha = sha256_file(fpath)
        if actual_sha != entry.get("sha256"):
            all_admitted = False
            unadmitted_reasons.append(
                f"sha256 mismatch for {fpath}: expected {entry.get('sha256')}, got {actual_sha}"
            )

    if not all_admitted:
        if not local_only:
            raise UnadmittedInputError("unadmitted_input: " + "; ".join(unadmitted_reasons))
        return False

    return not local_only


def load_bird_mini_dev(
    json_path: Path,
    dbs_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load the 500 BIRD-Dev SQLite items from BIRD Mini-Dev V2."""
    with json_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    records: list[dict[str, Any]] = []
    counts = {"total_loaded": 0, "missing_db": 0}
    db_hashes: dict[str, str] = {}

    for row in raw_items:
        db_id = str(row["db_id"])
        db_path = dbs_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            alt_path = dbs_dir / f"{db_id}.sqlite"
            if alt_path.exists():
                db_path = alt_path
            else:
                counts["missing_db"] += 1
                continue
        if db_id not in db_hashes:
            db_hashes[db_id] = sha256_file(db_path)

        qid = str(row.get("question_id", row.get("id", len(records))))
        question = str(row["question"])
        hint = row.get("evidence") or None
        if hint == "":
            hint = None
        sql = str(row["SQL"])
        gid = compute_group_id(db_id, question)
        records.append(
            {
                "item_ref": f"bird_minidev:{qid}",
                "group_id": gid,
                "split": assign_split(gid),
                "db_id": db_id,
                "db_path": db_path,
                "db_sha256": db_hashes[db_id],
                "question": question,
                "hint": hint,
                "gt_sql": sql,
                "difficulty": row.get("difficulty", "unknown"),
            }
        )
        counts["total_loaded"] += 1

    return records, counts


def load_flex_dataset(
    fp_jsonl: Path,
    fn_jsonl: Path,
    dbs_dir: Path,
    *,
    experts_csv: Path | None = None,
    snapshot: str = "flex",
) -> tuple[list[LabelledItem], dict[str, int]]:
    """Load the 200 FLEX expert-labelled BIRD Dev pairs (S4), excluding FP_consensus == 0.5."""
    expert_overrides: dict[str, str] = {}
    excluded_ids: set[str] = set()
    excluded_fp_indices: set[str] = set()
    if experts_csv is not None and experts_csv.exists():
        with experts_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cons = row.get("FP_consensus")
                if row.get("id") or row.get("question_id"):
                    key = str(row.get("id") or row.get("question_id") or "")
                    if cons is not None and str(cons).strip() == "0.5":
                        excluded_ids.add(key)
                    elif row.get("label"):
                        expert_overrides[key] = str(row["label"]).strip().lower()
                elif row.get("Index") is not None:
                    fp_idx = str(row["Index"]).strip()
                    if cons is not None and str(cons).strip() == "0.5":
                        excluded_fp_indices.add(fp_idx)

    items: list[LabelledItem] = []
    stats = {"loaded": 0, "excluded_consensus_tie": 0, "missing_db": 0, "invalid_label": 0}
    db_hashes: dict[str, str] = {}
    seen_item_ids: dict[str, int] = {}

    for family, path in (("flex_fp", fp_jsonl), ("flex_fn", fn_jsonl)):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_id = str(
                    row.get("id")
                    or row.get("question_id")
                    or (row["index"] if "index" in row else idx)
                )
                if (
                    row_id in excluded_ids
                    or (family == "flex_fp" and str(idx) in excluded_fp_indices)
                    or str(row.get("FP_consensus", "")).strip() == "0.5"
                ):
                    stats["excluded_consensus_tie"] += 1
                    continue

                raw_db_id = row.get("db_id")
                if not raw_db_id:
                    schema_text = str(row.get("schema", ""))
                    m_db = re.match(r"^\s*Database:\s*([^\s\n]+)", schema_text)
                    raw_db_id = m_db.group(1) if m_db else ""
                db_id = str(raw_db_id)
                if not db_id:
                    stats["missing_db"] += 1
                    continue
                db_path = dbs_dir / db_id / f"{db_id}.sqlite"
                if not db_path.exists():
                    alt = dbs_dir / f"{db_id}.sqlite"
                    if alt.exists():
                        db_path = alt
                    else:
                        stats["missing_db"] += 1
                        continue
                if db_id not in db_hashes:
                    db_hashes[db_id] = sha256_file(db_path)

                question = str(row["question"])
                hint = row.get("evidence") or row.get("hint") or None
                if hint == "":
                    hint = None
                gt_sql = str(row["gt_sql"])
                pred_sql = str(row["pred_sql"])
                raw_judge = row.get("human_judge")
                label: Literal["correct", "incorrect", "ambiguous", "gt_wrong"]
                if row_id in expert_overrides:
                    lbl_raw = expert_overrides[row_id]
                    if lbl_raw in ("1", "1.0", "true", "correct", "yes"):
                        label = "correct"
                    elif lbl_raw in ("0", "0.0", "false", "incorrect", "no"):
                        label = "incorrect"
                    else:
                        stats["invalid_label"] += 1
                        continue
                elif isinstance(raw_judge, (int, float)) and not isinstance(raw_judge, bool):
                    label = "correct" if float(raw_judge) >= 0.5 else "incorrect"
                else:
                    lbl_str = str(raw_judge).strip().lower() if raw_judge is not None else ""
                    if lbl_str in ("1", "1.0", "true", "correct", "yes"):
                        label = "correct"
                    elif lbl_str in ("0", "0.0", "false", "incorrect", "no"):
                        label = "incorrect"
                    else:
                        stats["invalid_label"] += 1
                        continue

                gid = compute_group_id(db_id, question)
                split = assign_split(gid)
                base_item_id = f"s4:{family}:{row_id}"
                occ = seen_item_ids.get(base_item_id, 0)
                seen_item_ids[base_item_id] = occ + 1
                item_id = base_item_id if occ == 0 else f"{base_item_id}:{occ}"
                items.append(
                    LabelledItem(
                        item_id=item_id,
                        group_id=gid,
                        split=split,
                        set="S4",
                        family=family,
                        slices=[],
                        trace=Trace(
                            trace_id=item_id,
                            group_id=gid,
                            question=question,
                            hint=hint,
                            db_id=db_id,
                            db_sha256=db_hashes[db_id],
                            steps=[Step(index=0, sql=pred_sql)],
                            final_answer=None,
                        ),
                        gt=GroundTruth(
                            item_ref=f"flex:{family}:{row_id}",
                            sql_chain=[gt_sql],
                        ),
                        label=label,
                        snapshot=snapshot,
                    )
                )
                stats["loaded"] += 1

    assert_disjoint_splits(items)
    return items, stats


def download_datasets(
    data_dir: Path,
    manifest_path: Path | None = None,
    *,
    minidev_db_url: str | None = DEFAULT_MINIDEV_DB_URL,
) -> dict[str, Any]:
    """Download pinned BIRD Mini-Dev V2 JSON, FLEX annotations, and optional DB zip into *data_dir*."""
    import httpx

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or (data_dir / "manifest.json")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"files": {}}

    targets = {
        raw_dir / "mini_dev_sqlite.json": (
            f"https://huggingface.co/datasets/{BIRD_MINIDEV_REPO}/resolve/"
            f"{BIRD_MINIDEV_REVISION}/{BIRD_MINIDEV_FILE}"
        ),
        raw_dir / "flex_fp.jsonl": (
            f"https://raw.githubusercontent.com/HeegyuKim/FLEX/{FLEX_COMMIT}/results/agreement/fp.jsonl"
        ),
        raw_dir / "flex_fn.jsonl": (
            f"https://raw.githubusercontent.com/HeegyuKim/FLEX/{FLEX_COMMIT}/results/agreement/fn.jsonl"
        ),
        raw_dir / "experts_agreement_200.csv": (
            f"https://raw.githubusercontent.com/HeegyuKim/FLEX/{FLEX_COMMIT}/results/agreement/experts_agreement_200.csv"
        ),
    }

    timeout = httpx.Timeout(600.0, connect=30.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for dest, url in targets.items():
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            digest = record_manifest_entry(
                manifest_path, dest, "pinned_download", base_dir=data_dir.parent
            )
            summary["files"][str(dest)] = digest

        if minidev_db_url:
            zip_path = raw_dir / "minidev.zip"
            with client.stream("GET", minidev_db_url) as stream:
                stream.raise_for_status()
                with zip_path.open("wb") as f:
                    for chunk in stream.iter_bytes(1024 * 1024):
                        f.write(chunk)
            record_manifest_entry(
                manifest_path, zip_path, "pinned_download", base_dir=data_dir.parent
            )
            dbs_dir = data_dir / "dbs"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dbs_dir)
            db_ids: list[str] = []
            for sqlite_file in sorted(dbs_dir.rglob("*.sqlite")):
                if "__MACOSX" in sqlite_file.parts or sqlite_file.name.startswith("._"):
                    continue
                db_id = sqlite_file.stem
                target = dbs_dir / db_id / f"{db_id}.sqlite"
                if sqlite_file.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sqlite_file.replace(target)
                    sqlite_file = target
                record_database_entry(manifest_path, db_id, sqlite_file, snapshot="minidev")
                db_ids.append(db_id)
            summary["databases"] = sorted(set(db_ids))

    return summary
