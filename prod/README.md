# Handover Production API

Standalone FastAPI service for the compiled GPT-5.2 handover programs.

This folder is intentionally self-contained:
- It does not import anything from the repository's top-level `src/`.
- It only reads model artifacts from the sibling [`compiled_programs/`](/Users/ac/handover-annotation/compiled_programs) directory by default.

## Default compiled programs

- SBAR: `sbar_span_gpt5-2_consensus.json`
- Checklist: `checklist_gpt_5-2_consensus.json`
- Unknown fact: `unknown_fact_binary_gpt5-2_user2_v2.json`

The unknown-fact default uses the newer GPT-5.2 `v2` artifact. Override any path with env vars if needed.

## Run

From [`/Users/ac/handover-annotation/prod`](/Users/ac/handover-annotation/prod):

```bash
uv sync
export OPENAI_API_KEY=your_key_here
uv run handover-prod
```

This runtime is pinned to Python 3.12 and `dspy-ai==3.0.4` to match the saved compiled-program artifacts.

Or with `uvicorn` directly:

```bash
uv run uvicorn handover_prod.app:app --host 0.0.0.0 --port 8000
```

## Configuration

- `OPENAI_API_KEY`: required for inference against OpenAI.
- `HANDOVER_MODEL`: defaults to `openai/gpt-5.2`
- `HANDOVER_REASONING_EFFORT`: optional OpenAI reasoning effort
- `HANDOVER_REQUEST_TIMEOUT_SECONDS`: defaults to `120`
- `HANDOVER_COMPILED_PROGRAMS_DIR`: defaults to the repo's [`compiled_programs/`](/Users/ac/handover-annotation/compiled_programs)
- `HANDOVER_SBAR_PROGRAM`: defaults to `sbar_span_gpt5-2_consensus.json`
- `HANDOVER_CHECKLIST_PROGRAM`: defaults to `checklist_gpt_5-2_consensus.json`
- `HANDOVER_UNKNOWN_FACT_PROGRAM`: defaults to `unknown_fact_binary_gpt5-2_user2_v2.json`

`HANDOVER_SBAR_PROGRAM`, `HANDOVER_CHECKLIST_PROGRAM`, and `HANDOVER_UNKNOWN_FACT_PROGRAM` can be either:
- an absolute path
- a filename inside `HANDOVER_COMPILED_PROGRAMS_DIR`
- a relative path from the current working directory

## Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /predict`
- `POST /predict/checklist`
- `POST /predict/sbar`
- `POST /predict/unknown-fact`

## Example

```bash
curl -X POST http://127.0.0.1:8000/predict/sbar \
  -H 'content-type: application/json' \
  -d '{
    "text": "[OUTGOING_NURSE] Day one post-op. [INCOMING_NURSE] Plan is to keep an eye on the wound."
  }'
```

Generic endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{
    "task": "checklist",
    "text": "[OUTGOING_NURSE] He came in with pneumonia and has no known allergies."
  }'
```
