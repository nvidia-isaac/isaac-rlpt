# isaac-rlpt-add-algorithm benchmark

## What is measured

Discovery: how often an agent loads this skill on its own for a prompt that needs it, and how often it loads it for a prompt that does not.
Each query in `evals/evals.json` (`trigger_evals`) runs 3 times in a scratch copy of the repository, with this skill as the only project skill; a query counts as triggered when the agent calls the Skill tool on this skill in at least 2 of the 3 runs, within 90 seconds.
The task evaluations in `evals/evals.json` (`evals`), which grade a full run against its expectations, have not been run yet.

## Results

Measured with Claude Code.

| Positive queries triggered | Positive runs | False triggers | Negative runs triggered |
|---:|---:|---:|---:|
| 4 of 4 | 12 of 12 | 0 of 4 | 0 of 12 |

| Query | Should trigger | Triggered |
|---|:---:|---:|
| add a new RL method that steers the latent like DSRL but uses TD3 | yes | 3/3 |
| how do I register a new algorithm so algorithm=myalgo works? | yes | 3/3 |
| write an AlgorithmSpec for an offline method that trains the full chunk | yes | 3/3 |
| my new algorithm's yaml fails the dry run with an undeclared key, what's wrong? | yes | 3/3 |
| set the DSRL entropy coefficient to 0.05 | no | 0/3 |
| add a new simulator for my robot | no | 0/3 |
| add OpenVLA as a base model | no | 0/3 |
| compare RFS and SAC-edit results on G1 | no | 0/3 |

## Limitations

- The queries were written together with the description, so they may favor it; a held-out set written by someone else is the stronger test.
- Three runs per query resolve large differences only; one run more or less moves a query across the threshold.
- Codex was not measured.

## Reproduce

```bash
uv run python tools/skill_trigger_eval.py skills/isaac-rlpt-add-algorithm
```
