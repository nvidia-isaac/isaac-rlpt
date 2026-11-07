# isaac-rlpt-add-base-model benchmark

## What is measured

Discovery: how often an agent loads this skill on its own for a prompt that needs it, and how often it loads it for a prompt that does not.
Each query in `evals/evals.json` (`trigger_evals`) runs 3 times in a scratch copy of the repository, with this skill as the only project skill; a query counts as triggered when the agent calls the Skill tool on this skill in at least 2 of the 3 runs, within 90 seconds.
The task evaluations in `evals/evals.json` (`evals`), which grade a full run against its expectations, have not been run yet.

## Results

Measured with Claude Code.

| Positive queries triggered | Positive runs | False triggers | Negative runs triggered |
|---:|---:|---:|---:|
| 3 of 4 | 9 of 12 | 0 of 4 | 0 of 12 |

| Query | Should trigger | Triggered |
|---|:---:|---:|
| I want DSRL to steer OpenVLA instead of GR00T, how do I add it as a base model? | yes | 3/3 |
| implement a BaseModelPolicy for our in-house flow policy and register it | yes | 3/3 |
| get_base_model('octo') raises unknown base model, how do I add octo? | yes | 3/3 |
| add a PPO fine-tuning adapter for a new flow-matching policy | yes | 0/3 |
| point the GR00T base model at a different checkpoint directory | no | 0/3 |
| add my robot simulator to the repo | no | 0/3 |
| add a new off-policy algorithm to the registry | no | 0/3 |
| what success rate does pi0 get on LIBERO-90? | no | 0/3 |

## Limitations

- The queries were written together with the description, so they may favor it; a held-out set written by someone else is the stronger test.
- Three runs per query resolve large differences only; one run more or less moves a query across the threshold.
- Codex was not measured.

## Reproduce

```bash
uv run python tools/skill_trigger_eval.py skills/isaac-rlpt-add-base-model
```
