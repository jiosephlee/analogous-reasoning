# Handoff Notes

## What was built

- `build_prompts.py` creates templated prompt JSONL files under
  `data/templated/{valid,test}/{zeroshot,feature_cot}`.
- `predict_api.py` runs the 2x2 evaluation matrix:
  - `zeroshot/no_tools`
  - `zeroshot/v17_get_features_only`
  - `feature_cot/no_tools`
  - `feature_cot/v17_get_features_only`
- `trace_viewer.html` is copied into each result run and can load full
  `traces.jsonl` or sample trace JSON files.
- `feature_cot_instruction.txt` is the feature-only CoT instruction. It avoids
  unavailable SAR, neighbor, analog, or retrieval instructions.
- `ENV_SETUP.md` documents conda setup, therapeutic-tools runtime envs, and GLM
  API checks.

## Prompt generation status

`python build_prompts.py --splits valid test` was run successfully.

Generated row counts:

| Split | Variant | Rows |
|---|---:|---:|
| valid | zeroshot | 2,114 |
| valid | feature_cot | 2,114 |
| test | zeroshot | 4,260 |
| test | feature_cot | 4,260 |

The raw task `SARSCoV2_3CLPro_Diamond` now has a direct prompt key in
`prompts.json`. The code also keeps a defensive alias map from
`SARSCoV2_3CLPro_Diamond` to `SARSCOV2_3CLPro_Diamond`.

## Local environment notes

The current shell is not the final harness env. `python predict_api.py
--preflight` correctly reports missing:

- `duckdb`
- `rdkit`
- editable `therapeutic_tools`
- `THERAPEUTIC_TOOLS_OPENRLHF_PYTHON`

Follow `ENV_SETUP.md` to create:

- `./conda_env/harness`
- `./conda_env/openrlhf`
- `./conda_env/minimol`

Install `therapeutic-tools` editable into all three envs.

## GLM / LiteLLM API notes

The local LiteLLM key is documented in `AGENTS.md`. Do not commit or paste that
raw key into result files. The runner can use it without echoing it:

```bash
python predict_api.py --api-key-from-agents --list-models
python predict_api.py --api-key-from-agents --api-smoke
```

Observed model list includes:

- `zai-org/GLM-5.2-FP8`
- `nvidia/GLM-5.1-NVFP4`
- `openai/gpt-oss-20b`

LiteLLM metadata says `zai-org/GLM-5.2-FP8` is configured as:

```text
provider:      hosted_vllm
mode:          chat
litellm model: hosted_vllm/zai-org/GLM-5.2-FP8
api_base:      https://vllm-zai-glm-5-2-locked-runai-test.inference.betty.parcc.upenn.edu/v1
backend id:    7aafcc80-2a4a-48d4-a7f5-a981bd916016
```

Current behavior from this machine:

- `/v1/models` succeeds and lists `zai-org/GLM-5.2-FP8`.
- `openai/gpt-oss-20b` chat smoke succeeds, proving the key and chat route work.
- `zai-org/GLM-5.2-FP8` chat smoke currently returns a hosted-vLLM 404, then
  LiteLLM places the deployment on cooldown.

The user provided a working curl example for the same GLM model, so the GLM
deployment appears intermittently healthy or environment-dependent rather than
misnamed.

## Verified commands

```bash
python -m py_compile build_prompts.py predict_api.py
python -m json.tool prompts.json >/dev/null
python build_prompts.py --splits valid test
python predict_api.py --split valid --tasks AMES --max-examples 2 --dry-run --timestamp dryrun2
python predict_api.py --api-key-from-agents --list-models
python predict_api.py --api-key-from-agents --model openai/gpt-oss-20b --api-smoke --timeout 90 --retries 0
```

## Next steps

1. Build the conda envs from `ENV_SETUP.md`.
2. Re-run `python predict_api.py --preflight` in the harness env.
3. Re-run `python predict_api.py --api-key-from-agents --api-smoke` once the GLM
   backend is healthy.
4. Run a one-example 2x2 live smoke:

```bash
python predict_api.py --api-key-from-agents --split valid --tasks AMES --max-examples 1 --rpm 30
```

5. Run the full test split when both API and therapeutic-tools preflights pass.
