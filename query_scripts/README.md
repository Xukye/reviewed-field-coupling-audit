# Query scripts

These are the scripts that sent the prompts to the model providers. They are released to document exactly how the models were called; they cannot run from this package alone, because they read the de-identified case file (`案件_165案_语义修复版.json`), which is built from licensed PKULaw judgments and is not redistributed (see `../data/README.md`). The instruction and input templates they use are identical to `../prompts/prompt_templates.json`.

API keys are never stored: each script reads the key from the `RUN_KEY` environment variable or asks for it at run time.

| Script | What it ran |
|---|---|
| `run_anchor_*.py`, `run_blind_*.py`, `run_anchor_zero_mimo.py` | Earlier per-provider runners for the true-anchor/pseudo-anchor and blind conditions (DeepSeek, Gemini, GLM, MiMo endpoints), with reasoning disabled (`thinking: disabled`; Gemini `thinking_budget=0`) and temperature 0. |
| `run_all.py` | Combined runner (used for DeepSeek-v4-pro, Qwen3.7-plus and Grok-4.3): blind, true-anchor and pseudo-anchor conditions for one model per invocation, with the model, endpoint and reasoning switch set by environment variables (`RUN_MODEL`, `RUN_BASE`, `RUN_EXTRA`). Default `RUN_EXTRA` is `{"thinking": {"type": "disabled"}}`; Grok via OpenRouter used `{"reasoning": {"enabled": false}}`. Temperature 0. |
| `rerun7_*.py` | 2026-06-30 re-queries of seven cases with corrected first-instance awards, for Gemini-2.5, GLM-5.1, MiMo and DeepSeek-Flash. |
| `rerun6_*.py` | 2026-10-15 re-queries of the six non-zero corrected cases (true anchor and four pseudo-anchors) for DeepSeek-v4-pro, Qwen3.7-plus and Grok-4.3. |
| `rerun1z_*.py` | 2026-10-15 re-query of CLI.C.575835292 (corrected first-instance award 0, true anchor) for the same three models. |

Results were merged into `../raw_api_outputs/`; merged rows carry a `补跑` field naming the script batch.
