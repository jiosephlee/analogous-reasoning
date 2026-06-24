# Local Environment Setup

This repo uses one harness environment and two separate therapeutic-tools
runtime environments. Keep the runtime environments separate because MolGpKa
and MiniMol pull different compiled chemistry and PyTorch stacks.

## Harness Environment

```bash
conda create -p ./conda_env/harness python=3.11 -y
conda activate ./conda_env/harness
conda install -c conda-forge rdkit duckdb -y
pip install httpx
pip install -e ./therapeutic-tools
```

## Therapeutic-Tools Runtime Environments

Follow the detailed install notes in `therapeutic-tools/envs/README.md`, then
install this checkout into both environments:

```bash
conda activate ./conda_env/openrlhf
pip install -e ./therapeutic-tools

conda activate ./conda_env/minimol
pip install -e ./therapeutic-tools
```

Point the harness at both runtime interpreters:

```bash
export THERAPEUTIC_TOOLS_OPENRLHF_PYTHON="$PWD/conda_env/openrlhf/bin/python"
export THERAPEUTIC_TOOLS_MINIMOL_PYTHON="$PWD/conda_env/minimol/bin/python"
```

The default feature groups do not use MiniMol:

```bash
molecular_profile structure_and_topology alert_screening
```

If you include `ionization_and_solubility`, the MiniMol runtime must pass
preflight.

## GLM API Setup And Smoke Test

The runner uses the OpenAI-compatible LiteLLM endpoint.

```bash
export OPENAI_BASE_URL="https://litellm.parcc.upenn.edu/v1"
export OPENAI_API_KEY="..."
export MODEL="zai-org/GLM-5.2-FP8"
python predict_api.py --api-smoke
```

For this workspace, `AGENTS.md` also documents the local LiteLLM key. To use it
without copying the key into your shell history:

```bash
python predict_api.py --api-key-from-agents --list-models
python predict_api.py --api-key-from-agents --api-smoke
```

If `--api-smoke` fails with a model error, inspect `results/models.json` and set
`MODEL` or pass `--model` to one of the model ids returned by `--list-models`.

Equivalent curl check:

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL:-zai-org/GLM-5.2-FP8}"'",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

The smoke test writes `results/api_smoke.json` and records only a short API key
fingerprint, never the raw key.

## Common Commands

Build templated prompts:

```bash
python build_prompts.py --splits valid test
```

Check local tool readiness:

```bash
python predict_api.py --preflight
```

Dry-run the full 2x2 matrix without API calls:

```bash
python predict_api.py --split valid --tasks AMES --max-examples 2 --dry-run
```

Run a one-example live smoke:

```bash
python predict_api.py --split valid --tasks AMES --max-examples 1 --rpm 30
```
