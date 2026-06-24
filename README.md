# analogous-reasoning

Evaluation harness for GLM 5.2 molecular-property prompting experiments.

## Quick start

Set up the local conda environments and GLM API variables with
`ENV_SETUP.md`.

Build templated prompts:

```bash
python build_prompts.py --splits valid test
```

Check GLM API connectivity after exporting `OPENAI_API_KEY`:

```bash
python predict_api.py --api-smoke
```

Dry-run the 2x2 matrix without API calls:

```bash
python predict_api.py --split valid --tasks AMES --max-examples 2 --dry-run
```

Run a small live smoke:

```bash
python predict_api.py --split valid --tasks AMES --max-examples 1 --rpm 30
```
