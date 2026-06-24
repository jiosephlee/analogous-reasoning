#!/usr/bin/env python3
"""Run GLM API evaluations over templated molecular-property prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from build_prompts import PROMPT_VARIANTS, TEMPLATED_DIR, build_prompts


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
VIEWER_TEMPLATE = ROOT / "trace_viewer.html"

TOOL_MODES = ("no_tools", "v17_get_features_only")
DEFAULT_FEATURE_GROUPS = ("molecular_profile", "structure_and_topology", "alert_screening")
ALL_FEATURE_GROUPS = (
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
)
MINIMOL_FEATURE_GROUPS = {"ionization_and_solubility"}


class RunError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
    }


def key_fingerprint(key: str | None) -> str | None:
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def command_output(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def api_config(args: argparse.Namespace) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    return {
        "model": args.model or os.environ.get("MODEL") or "zai-org/GLM-5.2-FP8",
        "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://litellm.parcc.upenn.edu/v1",
        "api_key_env": "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else "LITELLM_API_KEY",
        "api_key_present": bool(key),
        "api_key_fingerprint": key_fingerprint(key),
    }


def load_api_key_from_agents() -> str | None:
    agents_path = ROOT / "AGENTS.md"
    if not agents_path.exists():
        return None
    text = agents_path.read_text(encoding="utf-8")
    match = re.search(r'api key\s+"([^"]+)"', text, re.IGNORECASE)
    return match.group(1) if match else None


def apply_agents_key_if_requested(args: argparse.Namespace) -> None:
    if not getattr(args, "api_key_from_agents", False):
        return
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY"):
        return
    key = load_api_key_from_agents()
    if key:
        os.environ["OPENAI_API_KEY"] = key


def environment_snapshot(args: argparse.Namespace, feature_groups: list[str]) -> dict[str, Any]:
    cfg = api_config(args)
    cache_path = therapeutic_cache_path()
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
        "cwd": str(ROOT),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "api": cfg,
        "therapeutic_tools": {
            "openrlhf_python": os.environ.get("THERAPEUTIC_TOOLS_OPENRLHF_PYTHON"),
            "minimol_python": os.environ.get("THERAPEUTIC_TOOLS_MINIMOL_PYTHON"),
            "cache_backend": os.environ.get("THERAPEUTIC_TOOLS_CACHE_BACKEND"),
            "duckdb_cache": str(cache_path) if cache_path else None,
            "duckdb_cache_stat": file_stat(cache_path),
            "feature_groups": feature_groups,
        },
    }


def therapeutic_cache_path() -> Path | None:
    for name in (
        "THERAPEUTIC_TOOLS_DUCKDB_CACHE",
        "THERAPEUTIC_TOOLS_RUNTIME_CACHE_DUCKDB",
        "THERAPEUTIC_TOOLS_FEATURE_CACHE_DUCKDB",
    ):
        if os.environ.get(name):
            return Path(os.environ[name]).expanduser()
    return ROOT / "therapeutic-tools" / "cache" / "therapeutic_tools.duckdb"


def import_check(module: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module)
    except Exception as exc:
        return {"module": module, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"module": module, "ok": spec is not None, "origin": spec.origin if spec else None}


def runtime_import_check(python: str | None, modules: list[str]) -> dict[str, Any]:
    if not python:
        return {"python": python, "ok": False, "error": "interpreter env var is not set", "modules": modules}
    exe = Path(python).expanduser()
    if not exe.exists():
        return {"python": str(exe), "ok": False, "error": "interpreter does not exist", "modules": modules}
    code = (
        "import importlib.util,json;"
        f"mods={modules!r};"
        "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}, sort_keys=True))"
    )
    try:
        proc = subprocess.run([str(exe), "-c", code], capture_output=True, text=True, timeout=60, check=False)
    except Exception as exc:
        return {"python": str(exe), "ok": False, "error": f"{type(exc).__name__}: {exc}", "modules": modules}
    if proc.returncode != 0:
        return {"python": str(exe), "ok": False, "error": proc.stderr.strip() or proc.stdout.strip(), "modules": modules}
    found = json.loads(proc.stdout or "{}")
    missing = [m for m, ok in found.items() if not ok]
    return {"python": str(exe), "ok": not missing, "found": found, "missing": missing}


def preflight(feature_groups: list[str], require_api_key: bool = False) -> dict[str, Any]:
    harness_modules = ["httpx", "duckdb", "rdkit", "therapeutic_tools", "therapeutic_tools.tools.v17_get_features_only"]
    harness = [import_check(m) for m in harness_modules]
    openrlhf = runtime_import_check(
        os.environ.get("THERAPEUTIC_TOOLS_OPENRLHF_PYTHON"),
        ["rdkit", "molgpka", "therapeutic_tools"],
    )
    needs_minimol = bool(MINIMOL_FEATURE_GROUPS.intersection(feature_groups))
    minimol = None
    if needs_minimol:
        minimol = runtime_import_check(
            os.environ.get("THERAPEUTIC_TOOLS_MINIMOL_PYTHON"),
            ["rdkit", "minimol", "graphium", "torch_sparse", "therapeutic_tools"],
        )
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    status = {
        "harness": harness,
        "openrlhf": openrlhf,
        "minimol": minimol,
        "api_key_present": bool(api_key),
        "api_key_fingerprint": key_fingerprint(api_key),
        "feature_groups": feature_groups,
    }
    errors = []
    missing_harness = [item["module"] for item in harness if not item["ok"]]
    if missing_harness:
        errors.append(f"harness env missing modules: {', '.join(missing_harness)}")
    if not openrlhf["ok"]:
        errors.append(f"openrlhf runtime is not ready: {openrlhf}")
    if minimol is not None and not minimol["ok"]:
        errors.append(f"minimol runtime is not ready: {minimol}")
    if require_api_key and not api_key:
        errors.append("OPENAI_API_KEY or LITELLM_API_KEY is required")
    status["ok"] = not errors
    status["errors"] = errors
    return status


def require_preflight(feature_groups: list[str], require_api_key: bool = False) -> dict[str, Any]:
    status = preflight(feature_groups, require_api_key=require_api_key)
    if not status["ok"]:
        raise RunError("preflight failed:\n" + "\n".join(f"- {e}" for e in status["errors"]))
    return status


def load_examples(split: str, prompt_variant: str, tasks: list[str] | None, max_examples: int | None) -> list[dict[str, Any]]:
    split_dir = TEMPLATED_DIR / split / prompt_variant
    if not split_dir.exists():
        build_prompts([split], [prompt_variant])
    paths = sorted(split_dir.glob("*.jsonl"))
    if tasks:
        wanted = set(tasks)
        paths = [p for p in paths if p.stem in wanted]
    examples = []
    for path in paths:
        count = 0
        for row in read_jsonl(path):
            if max_examples is not None and count >= max_examples:
                break
            examples.append(row)
            count += 1
    if not examples:
        raise RunError(f"no examples found for split={split!r}, prompt_variant={prompt_variant!r}, tasks={tasks!r}")
    return examples


def parse_feature_groups(raw: list[str]) -> list[str]:
    if len(raw) == 1 and raw[0] == "all":
        return list(ALL_FEATURE_GROUPS)
    groups = []
    valid = set(ALL_FEATURE_GROUPS)
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if part not in valid:
                raise RunError(f"unknown feature group {part!r}; valid={sorted(valid)} or all")
            if part not in groups:
                groups.append(part)
    return groups or list(DEFAULT_FEATURE_GROUPS)


def get_feature_tool():
    from therapeutic_tools.tools.v17_get_features_only import GET_FEATURES_TOOL, get_features

    return GET_FEATURES_TOOL, get_features


def is_tool_error(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("Error:") or stripped.startswith("get_features: Error")


def compute_feature_text(smiles: str, feature_groups: list[str]) -> tuple[str | None, dict[str, Any]]:
    tool_schema, get_features = get_feature_tool()
    start = time.time()
    event = {
        "name": "get_features",
        "schema": tool_schema,
        "arguments": {"smiles": smiles, "feature_names": feature_groups},
        "started_at": utc_now(),
        "status": "ok",
    }
    try:
        output = get_features(smiles, feature_groups)
        event["output"] = output
        if not isinstance(output, str) or is_tool_error(output):
            event["status"] = "error"
            event["error"] = str(output)
            return None, event
        return output, event
    except Exception as exc:
        event["status"] = "error"
        event["error"] = f"{type(exc).__name__}: {exc}"
        return None, event
    finally:
        event["ended_at"] = utc_now()
        event["latency_ms"] = round((time.time() - start) * 1000)


def inject_feature_evidence(prompt: str, feature_text: str) -> str:
    evidence = (
        "Therapeutic feature evidence for the molecule:\n"
        "```\n"
        f"{feature_text.strip()}\n"
        "```"
    )
    marker = "Answer:"
    idx = prompt.rfind(marker)
    if idx < 0:
        return f"{prompt.rstrip()}\n\n{evidence}\n\nAnswer:"
    prefix = prompt[:idx].rstrip()
    suffix = prompt[idx:]
    return f"{prefix}\n\n{evidence}\n\n{suffix}"


def warm_feature_cache(
    examples: list[dict[str, Any]],
    feature_groups: list[str],
    fail_fast: bool,
) -> tuple[dict[str, str], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    unique_smiles = []
    seen = set()
    for row in examples:
        smiles = row["smiles"]
        if smiles not in seen:
            unique_smiles.append(smiles)
            seen.add(smiles)
    feature_by_smiles: dict[str, str] = {}
    events_by_smiles: dict[str, list[dict[str, Any]]] = {}
    errors = []
    start = time.time()
    for smiles in unique_smiles:
        text, event = compute_feature_text(smiles, feature_groups)
        events_by_smiles[smiles] = [event]
        if text is None:
            errors.append({"smiles": smiles, "error": event.get("error")})
            if fail_fast:
                raise RunError(f"feature warmup failed for {smiles!r}: {event.get('error')}")
        else:
            feature_by_smiles[smiles] = text
    summary = {
        "feature_groups": feature_groups,
        "unique_smiles": len(unique_smiles),
        "computed": len(feature_by_smiles),
        "errors": errors,
        "latency_ms": round((time.time() - start) * 1000),
        "cache_path": str(therapeutic_cache_path()) if therapeutic_cache_path() else None,
    }
    return feature_by_smiles, summary, events_by_smiles


def parse_answer(text: str) -> tuple[int | None, str | None]:
    answer_parts = re.split(r"Answer\s*:", text, flags=re.IGNORECASE)
    search_area = answer_parts[-1] if len(answer_parts) > 1 else text
    matches = re.findall(r"\(([AB])\)|\b([AB])\b", search_area)
    letters = [a or b for a, b in matches]
    if not letters:
        return None, "could not find final A/B answer"
    letter = letters[-1]
    return (0 if letter == "A" else 1), None


def binary_macro_f1(labels: list[int], preds: list[int | None]) -> float:
    f1s = []
    for cls in (0, 1):
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        denom = (2 * tp) + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(f1s) / len(f1s)


def score_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = [r for r in rows if "label" in r]
    total = len(scored)
    correct = sum(1 for r in scored if r.get("correct") is True)
    labels = [int(r["label"]) for r in scored]
    preds = [r.get("prediction") for r in scored]
    aggregate = {
        "n": total,
        "accuracy": (correct / total) if total else 0.0,
        "macro_f1": binary_macro_f1(labels, preds) if total else 0.0,
        "parse_failures": sum(1 for r in scored if r.get("prediction") is None),
        "zero_division": 0,
    }
    by_task: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[str(row["task"])].append(row)
    for task, task_rows in sorted(grouped.items()):
        t_labels = [int(r["label"]) for r in task_rows]
        t_preds = [r.get("prediction") for r in task_rows]
        t_correct = sum(1 for r in task_rows if r.get("correct") is True)
        by_task[task] = {
            "n": len(task_rows),
            "accuracy": t_correct / len(task_rows),
            "macro_f1": binary_macro_f1(t_labels, t_preds),
            "parse_failures": sum(1 for r in task_rows if r.get("prediction") is None),
            "label_distribution": {str(cls): t_labels.count(cls) for cls in (0, 1)},
        }
    return aggregate, by_task


def chat_completion(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], int, float]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
            latency = time.time() - start
            if response.status_code < 400:
                return response.json(), response.status_code, latency
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:
            latency = time.time() - start
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    raise RunError(last_error or "API request failed")


def api_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cfg = api_config(args)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "started_at": utc_now(),
            "ended_at": utc_now(),
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key_env": cfg["api_key_env"],
            "api_key_present": False,
            "error": "OPENAI_API_KEY or LITELLM_API_KEY is required",
        }
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 64,
    }
    started = utc_now()
    try:
        response, status_code, latency = chat_completion(cfg["base_url"], api_key, payload, args.timeout, args.retries)
    except Exception as exc:
        return {
            "ok": False,
            "started_at": started,
            "ended_at": utc_now(),
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key_env": cfg["api_key_env"],
            "api_key_fingerprint": cfg["api_key_fingerprint"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = extract_assistant_text(response)
    ok = bool(text.strip())
    return {
        "ok": ok,
        "started_at": started,
        "ended_at": utc_now(),
        "status_code": status_code,
        "latency_ms": round(latency * 1000),
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "api_key_env": cfg["api_key_env"],
        "api_key_fingerprint": cfg["api_key_fingerprint"],
        "response_id": response.get("id"),
        "token_usage": response.get("usage"),
        "assistant_text": text,
    }


def list_models(args: argparse.Namespace) -> dict[str, Any]:
    cfg = api_config(args)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "base_url": cfg["base_url"],
            "api_key_present": False,
            "error": "OPENAI_API_KEY or LITELLM_API_KEY is required",
        }
    url = cfg["base_url"].rstrip("/") + "/models"
    started = time.time()
    try:
        response = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=args.timeout)
        latency_ms = round((time.time() - started) * 1000)
        data = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "base_url": cfg["base_url"],
            "api_key_fingerprint": cfg["api_key_fingerprint"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    model_ids = [
        item.get("id")
        for item in data.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    return {
        "ok": response.status_code < 400,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "base_url": cfg["base_url"],
        "api_key_fingerprint": cfg["api_key_fingerprint"],
        "models": model_ids,
    }


def extract_assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def make_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a careful molecular property classifier. End with Answer: (A) or Answer: (B).",
        },
        {"role": "user", "content": prompt},
    ]


def maybe_sleep_for_rate_limit(last_request_at: float | None, rpm: float | None) -> float:
    if not rpm or rpm <= 0:
        return time.time()
    min_interval = 60.0 / rpm
    if last_request_at is not None:
        elapsed = time.time() - last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    return time.time()


def copy_viewer(run_dir: Path) -> None:
    if VIEWER_TEMPLATE.exists():
        shutil.copyfile(VIEWER_TEMPLATE, run_dir / "viewer.html")


def write_samples(run_dir: Path, traces: list[dict[str, Any]], max_samples: int) -> None:
    sample_dir = run_dir / "sample_traces"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("*.json"):
        old.unlink()
    for idx, trace in enumerate(traces[:max_samples]):
        json_dump(sample_dir / f"sample_{idx:03d}.json", trace)


def completed_ids(predictions_path: Path) -> set[str]:
    return {str(row.get("example_id")) for row in read_jsonl(predictions_path)}


def run_method(
    args: argparse.Namespace,
    prompt_variant: str,
    tool_mode: str,
    run_ts: str,
    feature_groups: list[str],
) -> Path:
    cfg = api_config(args)
    method = f"{prompt_variant}__{tool_mode}"
    run_dir = Path(args.output_root) / safe_name(cfg["model"]) / method / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.split, prompt_variant, args.tasks, args.max_examples)
    if args.cache_mode == "cache_required":
        cache_path = therapeutic_cache_path()
        if not cache_path or not cache_path.exists():
            raise RunError(f"cache_required selected but DuckDB cache is missing: {cache_path}")

    run_config = {
        "run_id": f"{safe_name(cfg['model'])}/{method}/{run_ts}",
        "created_at_utc": utc_now(),
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "split": args.split,
        "tasks": args.tasks,
        "max_examples": args.max_examples,
        "prompt_variant": prompt_variant,
        "tool_mode": tool_mode,
        "feature_groups": feature_groups if tool_mode != "no_tools" else [],
        "temperature": args.temperature,
        "timeout": args.timeout,
        "rpm": args.rpm,
        "cache_mode": args.cache_mode,
        "warm_cache": args.warm_cache,
        "tool_error_policy": args.tool_error_policy,
        "zero_division": 0,
        "api_key_env": cfg["api_key_env"],
        "api_key_fingerprint": cfg["api_key_fingerprint"],
    }
    json_dump(run_dir / "run_config.json", run_config)
    json_dump(run_dir / "environment.json", environment_snapshot(args, feature_groups))
    json_dump(run_dir / "task_aliases.json", {"SARSCoV2_3CLPro_Diamond": "SARSCOV2_3CLPro_Diamond"})
    copy_viewer(run_dir)

    if args.dry_run:
        planned = [{"example_id": e["example_id"], "task": e["task"], "row_index": e["row_index"]} for e in examples]
        json_dump(run_dir / "planned_examples.json", planned)
        json_dump(run_dir / "cache_summary.json", {"dry_run": True})
        return run_dir

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise RunError("OPENAI_API_KEY or LITELLM_API_KEY is required for live prediction")

    feature_by_smiles: dict[str, str] = {}
    feature_events_by_smiles: dict[str, list[dict[str, Any]]] = {}
    cache_summary: dict[str, Any] = {"tool_mode": tool_mode}
    if tool_mode != "no_tools":
        require_preflight(feature_groups, require_api_key=True)
        if args.warm_cache:
            feature_by_smiles, cache_summary, feature_events_by_smiles = warm_feature_cache(
                examples,
                feature_groups,
                fail_fast=args.tool_error_policy == "fail_fast",
            )
        else:
            cache_summary = {"warm_cache": False, "feature_groups": feature_groups}
    json_dump(run_dir / "cache_summary.json", cache_summary)

    predictions_path = run_dir / "predictions.jsonl"
    traces_path = run_dir / "traces.jsonl"
    done = completed_ids(predictions_path) if args.resume else set()
    last_request_at = None

    for example in examples:
        example_id = example["example_id"]
        if example_id in done:
            continue
        prompt = example["prompt"]
        tool_events = []
        feature_text = None
        failure = None
        if tool_mode != "no_tools":
            smiles = example["smiles"]
            feature_text = feature_by_smiles.get(smiles)
            if feature_text is None:
                text, event = compute_feature_text(smiles, feature_groups)
                tool_events.append(event)
                if text is None:
                    failure = {"stage": "tool", "message": event.get("error")}
                else:
                    feature_text = text
            else:
                tool_events.extend(feature_events_by_smiles.get(smiles, []))
            if feature_text is not None:
                prompt = inject_feature_evidence(prompt, feature_text)
        messages = make_messages(prompt)
        raw_response = None
        assistant_text = ""
        prediction = None
        parse_error = None
        latency_ms = None
        token_usage = None
        if failure is None:
            try:
                last_request_at = maybe_sleep_for_rate_limit(last_request_at, args.rpm)
                payload = {
                    "model": cfg["model"],
                    "messages": messages,
                    "temperature": args.temperature,
                }
                if args.max_tokens is not None:
                    payload["max_tokens"] = args.max_tokens
                raw_response, status_code, latency = chat_completion(
                    cfg["base_url"], api_key, payload, args.timeout, args.retries
                )
                latency_ms = round(latency * 1000)
                assistant_text = extract_assistant_text(raw_response)
                token_usage = raw_response.get("usage")
                prediction, parse_error = parse_answer(assistant_text)
                if parse_error:
                    failure = {"stage": "parse", "message": parse_error}
            except Exception as exc:
                failure = {"stage": "api", "message": f"{type(exc).__name__}: {exc}"}
        correct = prediction == example["label"] if prediction is not None else False
        pred_row = {
            "run_id": run_config["run_id"],
            "example_id": example_id,
            "task": example["task"],
            "split": example["split"],
            "row_index": example["row_index"],
            "smiles": example["smiles"],
            "label": example["label"],
            "prompt_variant": prompt_variant,
            "tool_mode": tool_mode,
            "model": cfg["model"],
            "prediction": prediction,
            "parsed_answer": None if prediction is None else ("A" if prediction == 0 else "B"),
            "correct": correct,
            "parse_error": parse_error,
            "failure": failure,
            "prompt_text_hash": sha256_text(prompt),
            "feature_text_hash": sha256_text(feature_text) if feature_text else None,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
        }
        trace = {
            **pred_row,
            "messages": messages,
            "tool_events": tool_events,
            "raw_response": raw_response,
            "assistant_text": assistant_text,
            "created_at_utc": utc_now(),
        }
        append_jsonl(predictions_path, pred_row)
        append_jsonl(traces_path, trace)

    predictions = read_jsonl(predictions_path)
    traces = read_jsonl(traces_path)
    scores, scores_by_task = score_predictions(predictions)
    json_dump(run_dir / "scores.json", scores)
    json_dump(run_dir / "scores_by_task.json", scores_by_task)
    write_samples(run_dir, traces, args.max_sample_traces)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test", "train"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum rows per task.")
    parser.add_argument("--prompt-variant", default="all", choices=["all", *PROMPT_VARIANTS])
    parser.add_argument("--tool-mode", default="all", choices=["all", *TOOL_MODES])
    parser.add_argument("--feature-groups", nargs="+", default=list(DEFAULT_FEATURE_GROUPS))
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-smoke", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--api-key-from-agents", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--build-prompts", action="store_true")
    parser.add_argument("--cache-mode", default="compute_if_missing", choices=["compute_if_missing", "cache_required"])
    parser.add_argument("--no-warm-cache", dest="warm_cache", action="store_false")
    parser.set_defaults(warm_cache=True)
    parser.add_argument("--tool-error-policy", default="fail_fast", choices=["fail_fast", "row_failure"])
    parser.add_argument("--max-sample-traces", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_agents_key_if_requested(args)
    feature_groups = parse_feature_groups(args.feature_groups)
    if args.build_prompts:
        build_prompts(["valid", "test"], PROMPT_VARIANTS)
    if args.api_smoke:
        result = api_smoke(args)
        out_dir = Path(args.output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_dump(out_dir / "api_smoke.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.list_models:
        result = list_models(args)
        out_dir = Path(args.output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_dump(out_dir / "models.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.preflight:
        status = preflight(feature_groups, require_api_key=False)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["ok"] else 1

    prompt_variants = list(PROMPT_VARIANTS) if args.prompt_variant == "all" else [args.prompt_variant]
    tool_modes = list(TOOL_MODES) if args.tool_mode == "all" else [args.tool_mode]
    if any(mode != "no_tools" for mode in tool_modes) and not args.dry_run:
        require_preflight(feature_groups, require_api_key=True)
    run_ts = args.timestamp or timestamp()
    run_dirs = []
    for prompt_variant in prompt_variants:
        for tool_mode in tool_modes:
            run_dirs.append(str(run_method(args, prompt_variant, tool_mode, run_ts, feature_groups)))
    print(json.dumps({"run_dirs": run_dirs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
