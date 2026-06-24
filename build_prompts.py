#!/usr/bin/env python3
"""Build templated valid/test prompt JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
TEMPLATED_DIR = ROOT / "data" / "templated"
PROMPTS_PATH = ROOT / "prompts.json"
FEATURE_COT_PATH = ROOT / "feature_cot_instruction.txt"

TASK_ALIASES = {
    "SARSCoV2_3CLPro_Diamond": "SARSCOV2_3CLPro_Diamond",
}

PROMPT_VARIANTS = ("zeroshot", "feature_cot")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(k): str(v) for k, v in prompts.items()}


def resolve_template(task: str, prompts: dict[str, str]) -> tuple[str, str]:
    if task in prompts:
        return task, prompts[task]
    alias = TASK_ALIASES.get(task)
    if alias and alias in prompts:
        return alias, prompts[alias]
    raise KeyError(f"missing prompt template for task {task!r}")


def render_base_prompt(template: str, smiles: str) -> str:
    if "{Drug SMILES}" not in template:
        raise ValueError("template does not contain {Drug SMILES}")
    return template.replace("{Drug SMILES}", smiles)


def add_feature_cot(base_prompt: str, instruction: str) -> str:
    marker = "Answer:"
    idx = base_prompt.rfind(marker)
    if idx < 0:
        return f"{base_prompt.rstrip()}\n\n{instruction.strip()}\nAnswer:"
    prefix = base_prompt[:idx].rstrip()
    return f"{prefix}\n\n{instruction.strip()}\n{marker}"


def iter_raw_rows(path: Path, max_examples: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_examples is not None and idx >= max_examples:
                break
            if not line.strip():
                continue
            yield idx, json.loads(line)


def build_split(
    split: str,
    variants: Iterable[str],
    prompts: dict[str, str],
    cot_instruction: str,
    max_examples: int | None = None,
) -> dict[str, Any]:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"raw split does not exist: {split_dir}")

    variants = tuple(variants)
    for variant in variants:
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown prompt variant {variant!r}; valid={PROMPT_VARIANTS}")

    summary: dict[str, Any] = {
        "split": split,
        "variants": list(variants),
        "tasks": {},
        "aliases": {},
    }
    for raw_path in sorted(split_dir.glob("*.jsonl")):
        task = raw_path.stem
        template_key, template = resolve_template(task, prompts)
        if template_key != task:
            summary["aliases"][task] = template_key
        template_hash = sha256_text(template)
        for variant in variants:
            out_dir = TEMPLATED_DIR / split / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / raw_path.name
            count = 0
            with out_path.open("w", encoding="utf-8") as out:
                for row_index, row in iter_raw_rows(raw_path, max_examples=max_examples):
                    smiles = row.get("drug")
                    label = row.get("Y")
                    if not isinstance(smiles, str):
                        raise ValueError(f"{raw_path}:{row_index + 1} missing string field 'drug'")
                    if label not in (0, 1):
                        raise ValueError(f"{raw_path}:{row_index + 1} has non-binary label {label!r}")
                    prompt = render_base_prompt(template, smiles)
                    cot_hash = None
                    if variant == "feature_cot":
                        prompt = add_feature_cot(prompt, cot_instruction)
                        cot_hash = sha256_text(cot_instruction)
                    record = {
                        "example_id": f"{split}:{task}:{row_index}",
                        "id": f"{split}:{task}:{row_index}",
                        "split": split,
                        "task": task,
                        "row_index": row_index,
                        "smiles": smiles,
                        "label": label,
                        "answer_letter": "A" if label == 0 else "B",
                        "prompt_variant": variant,
                        "prompt": prompt,
                        "prompt_text": prompt,
                        "template_key": template_key,
                        "template_source_sha256": template_hash,
                        "cot_instruction_sha256": cot_hash,
                    }
                    out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
            summary["tasks"].setdefault(task, {})[variant] = {
                "rows": count,
                "path": str(out_path),
                "template_key": template_key,
            }
    return summary


def build_prompts(
    splits: Iterable[str] = ("valid", "test"),
    variants: Iterable[str] = PROMPT_VARIANTS,
    max_examples: int | None = None,
) -> dict[str, Any]:
    prompts = load_prompts()
    cot_instruction = FEATURE_COT_PATH.read_text(encoding="utf-8")
    all_summary = {
        "prompts_path": str(PROMPTS_PATH),
        "feature_cot_instruction_path": str(FEATURE_COT_PATH),
        "prompt_variants": list(variants),
        "splits": {},
    }
    for split in splits:
        all_summary["splits"][split] = build_split(
            split=split,
            variants=variants,
            prompts=prompts,
            cot_instruction=cot_instruction,
            max_examples=max_examples,
        )
    TEMPLATED_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TEMPLATED_DIR / "manifest.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return all_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["valid", "test", "train"])
    parser.add_argument("--prompt-variants", nargs="+", default=list(PROMPT_VARIANTS), choices=list(PROMPT_VARIANTS))
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_prompts(args.splits, args.prompt_variants, args.max_examples)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
