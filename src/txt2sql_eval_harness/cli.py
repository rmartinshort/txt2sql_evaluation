"""Command-line interface for the Text2SQL Decision-Model Judge harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from txt2sql_eval_harness.codec import decode_labelled_item, encode_labelled_item
from txt2sql_eval_harness.data import (
    download_datasets,
    load_bird_mini_dev,
    load_flex_dataset,
    record_manifest_entry,
)
from txt2sql_eval_harness.generators import (
    generate_e5b_pairs,
    generate_s1_items,
    generate_s2_items,
    generate_s3_items,
    generate_s3m_items,
)
from txt2sql_eval_harness.metrics import load_records
from txt2sql_eval_harness.models import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_JEV_MODEL,
    EvalConfig,
    GroundTruth,
    LabelledItem,
    RecordedMethodName,
    ResultTable,
    Trace,
)
from txt2sql_eval_harness.reporting import generate_all_reports_and_figures
from txt2sql_eval_harness.runner import run_experiment, select_common_subset_c, tag_slices
from txt2sql_eval_harness.sandbox import execute_sandboxed

DEFAULT_METHODS: tuple[RecordedMethodName, ...] = (
    "H+E",
    "B6",
    "B6+E",
    "B4",
    "B1",
    "B2",
    "B-IBM-subset",
    "B-IBM-judge",
)


def _load_items_jsonl(path: Path) -> list[LabelledItem]:
    """Read a JSONL file of LabelledItem records into memory, skipping blank lines."""
    items: list[LabelledItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            items.append(decode_labelled_item(json.loads(s)))
    return items


def _save_items_jsonl(path: Path, items: list[LabelledItem]) -> None:
    """Write a sequence of LabelledItem records as compact JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            line = json.dumps(
                encode_labelled_item(it),
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            fh.write(line + "\n")


def _bird_records_to_items(
    records: list[dict[str, Any]],
    cfg: EvalConfig,
) -> list[LabelledItem]:
    """Convert loaded BIRD Mini-Dev dictionaries into LabelledItem seeds with sandboxed GT results."""
    items: list[LabelledItem] = []
    for rec in records:
        db_path: Path = rec["db_path"]
        gt_sql = str(rec["gt_sql"])
        gt_res, _err, _refs = execute_sandboxed(db_path, gt_sql, cfg)
        results: list[ResultTable] | None = [gt_res] if gt_res is not None else None
        item_id = str(rec["item_ref"])
        items.append(
            LabelledItem(
                item_id=item_id,
                group_id=str(rec["group_id"]),
                split=rec["split"],
                set="S1",
                family="base",
                slices=[],
                trace=Trace(
                    trace_id=item_id,
                    group_id=str(rec["group_id"]),
                    question=str(rec["question"]),
                    hint=rec.get("hint"),
                    db_id=str(rec["db_id"]),
                    db_sha256=str(rec["db_sha256"]),
                    steps=[],
                    final_answer=None,
                ),
                gt=GroundTruth(
                    item_ref=item_id,
                    sql_chain=[gt_sql],
                    results=results,
                ),
                label="correct",
                snapshot="minidev",
            )
        )
    return items


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint dispatching download, generate, run, and report subcommands."""
    parser = argparse.ArgumentParser(
        prog="txt2sql-judge",
        description="Text2SQL Decision-Model Judge Harness CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download BIRD Mini-Dev V2 and FLEX datasets.")
    p_dl.add_argument("--data-dir", type=Path, default=Path("data"))

    p_gen = sub.add_parser("generate", help="Generate S1, S2, S3, S3m, and E5b items locally.")
    p_gen.add_argument("--data-dir", type=Path, default=Path("data"))
    p_gen.add_argument("--out", type=Path, default=Path("data/items.jsonl"))
    p_gen.add_argument("--e5b-out", type=Path, default=Path("data/e5b_items.jsonl"))

    p_lc = sub.add_parser(
        "generate-langchain",
        help="Run LangChain SQL agent on benchmark questions to produce S5 evaluation items.",
    )
    p_lc.add_argument("--dataset", choices=["bird", "flex"], default="bird")
    p_lc.add_argument("--data-dir", type=Path, default=Path("data"))
    p_lc.add_argument("--model", type=str, default=DEFAULT_AGENT_MODEL)
    p_lc.add_argument("--temperature", type=float, default=0.0)
    p_lc.add_argument("--limit", type=int, default=20)
    p_lc.add_argument("--seed", type=int, default=42)
    p_lc.add_argument("--split", choices=["all", "calibration", "test"], default="test")
    p_lc.add_argument("--db-id", type=str, default=None)
    p_lc.add_argument("--max-iterations", type=int, default=50)
    p_lc.add_argument("--sql-only", action="store_true")
    p_lc.add_argument("--concurrency", type=int, default=8)
    p_lc.add_argument("--verbose", action="store_true")
    p_lc.add_argument("--out", type=Path, default=Path("data/langchain_items.jsonl"))

    p_run = sub.add_parser("run", help="Run evaluation methods over an items JSONL.")
    p_run.add_argument("--items", type=Path, required=True)
    p_run.add_argument("--root", type=Path, default=Path("experiments/default"))
    p_run.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    p_run.add_argument("--db-dir", type=Path, default=Path("data/dbs"))
    p_run.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    p_run.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p_run.add_argument("--local-only", action="store_true")
    p_run.add_argument("--cache-only", action="store_true")
    p_run.add_argument("--enable-d5", action="store_true")
    p_run.add_argument("--subset-c", action="store_true")
    p_run.add_argument("--cap-usd", type=float, default=150.0)
    p_run.add_argument("--decision-model", type=str, default=DEFAULT_JEV_MODEL)
    p_run.add_argument("--concurrency", type=int, default=8)
    p_run.add_argument("--verbose", action="store_true")

    p_rep = sub.add_parser("report", help="Generate T1-T6 tables and F2-F6 charts.")
    p_rep.add_argument("--items", type=Path, required=True)
    p_rep.add_argument("--records", type=Path, required=True)
    p_rep.add_argument("--out-dir", type=Path, default=Path("runs/report"))
    p_rep.add_argument("--spend", type=Path, default=None)
    p_rep.add_argument("--bootstrap-resamples", type=int, default=1000)

    args = parser.parse_args(argv)

    if args.command == "download":
        summary = download_datasets(args.data_dir, args.data_dir / "manifest.json")
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "generate":
        manifest = args.data_dir / "manifest.json"
        raw_dir = args.data_dir / "raw"
        db_dir = args.data_dir / "dbs"
        bird_records, _ = load_bird_mini_dev(raw_dir / "mini_dev_sqlite.json", db_dir)
        bird_items = _bird_records_to_items(bird_records, EvalConfig())
        flex_items, _ = load_flex_dataset(
            raw_dir / "flex_fp.jsonl",
            raw_dir / "flex_fn.jsonl",
            db_dir,
            experts_csv=raw_dir / "experts_agreement_200.csv",
        )

        s1 = generate_s1_items(bird_items, db_dir)
        s2 = generate_s2_items(bird_items, db_dir)
        s2_by_base: dict[str, list[LabelledItem]] = {}
        for m in s2.items:
            parts = m.item_id.split("__")
            if len(parts) >= 2:
                s2_by_base.setdefault(parts[1], []).append(m)
        s3 = generate_s3_items(bird_items, s2_by_base=s2_by_base)
        s3m = generate_s3m_items(
            [it for it in s3.items if it.family == "correct"],
            bird_items,
            s2_by_base=s2_by_base,
        )
        all_gen = tag_slices(s1.items + s2.items + s3.items + s3m.items + flex_items)
        e5b = generate_e5b_pairs(s3.items)
        _save_items_jsonl(args.out, all_gen)
        _save_items_jsonl(args.e5b_out, e5b.items)
        record_manifest_entry(manifest, args.out, "harness_derivative")
        record_manifest_entry(manifest, args.e5b_out, "harness_derivative")
        print(
            json.dumps(
                {
                    "generated": len(all_gen),
                    "e5b_items": len(e5b.items),
                    "discards": {
                        **s1.discarded,
                        **s2.discarded,
                        **s3.discarded,
                        **s3m.discarded,
                        **e5b.discarded,
                    },
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run":
        cfg = EvalConfig(enable_d5=bool(args.enable_d5), jev_model=args.decision_model)
        items = _load_items_jsonl(args.items)
        if args.subset_c:
            items = select_common_subset_c(items)
        res = run_experiment(
            items=items,
            methods=list(args.methods),  # type: ignore[arg-type]
            cfg=cfg,
            root=args.root,
            run_dir=args.run_dir,
            manifest_path=args.manifest,
            input_files=[args.items],
            db_dir=args.db_dir if args.db_dir.exists() else None,
            local_only=bool(args.local_only),
            cache_only=bool(args.cache_only),
            cap_usd=float(args.cap_usd),
            concurrency=int(args.concurrency),
            verbose=bool(args.verbose),
        )
        print(
            json.dumps(
                {
                    "run_dir": str(res.run_dir),
                    "written": res.written_count,
                    "skipped": res.skipped_existing_count,
                    "stopped_on_spend_cap": res.stopped_on_spend_cap,
                    "allow_remote": res.allow_remote,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "report":
        items = _load_items_jsonl(args.items)
        c_items = select_common_subset_c(items) or items
        records = load_records(args.records, include_h=True)
        out = generate_all_reports_and_figures(
            args.out_dir,
            items,
            c_items,
            records,
            spend_json_path=args.spend,
            bootstrap_resamples=int(args.bootstrap_resamples),
        )
        print(json.dumps({"tables": list(out.keys()), "out_dir": str(args.out_dir)}, indent=2))
        return 0

    if args.command == "generate-langchain":
        from txt2sql_eval_harness.langchain_agent import build_chat_model, generate_langchain_items

        manifest = args.data_dir / "manifest.json"
        raw_dir = args.data_dir / "raw"
        db_dir = args.data_dir / "dbs"
        cfg = EvalConfig()

        if args.dataset == "bird":
            bird_records, _ = load_bird_mini_dev(raw_dir / "mini_dev_sqlite.json", db_dir)
            base_items = _bird_records_to_items(bird_records, cfg)
        else:
            base_items, _ = load_flex_dataset(
                raw_dir / "flex_fp.jsonl",
                raw_dir / "flex_fn.jsonl",
                db_dir,
                experts_csv=raw_dir / "experts_agreement_200.csv",
            )

        llm = build_chat_model(model=args.model, temperature=float(args.temperature))
        _, stats = generate_langchain_items(
            base_items,
            db_dir,
            llm,
            args.model,
            cfg=cfg,
            limit=int(args.limit),
            seed=int(args.seed),
            split=args.split,
            db_id=args.db_id,
            max_iterations=int(args.max_iterations),
            sql_only=bool(args.sql_only),
            concurrency=int(args.concurrency),
            verbose=bool(args.verbose),
            out_path=args.out,
        )
        if manifest.exists():
            record_manifest_entry(manifest, args.out, "harness_derivative")
        print(json.dumps({**stats, "out": str(args.out)}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
