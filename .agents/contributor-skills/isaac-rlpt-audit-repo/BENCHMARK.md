# isaac-rlpt-audit-repo benchmark

## What is measured

Discovery: how often an agent loads this skill on its own for a prompt that needs it, and how often it loads it for a prompt that does not.
Each query in `evals/evals.json` (`trigger_evals`) runs 3 times in a scratch copy of the repository, with this skill as the only project skill; a query counts as triggered when the agent calls the Skill tool on this skill in at least 2 of the 3 runs, within 90 seconds.
The task evaluations in `evals/evals.json` (`evals`), which grade a full run against its expectations, have not been run yet.

## Results

Measured with Claude Code.

| Positive queries triggered | Positive runs | False triggers | Negative runs triggered |
|---:|---:|---:|---:|
| 3 of 3 | 9 of 9 | 0 of 3 | 0 of 9 |

| Query | Should trigger | Triggered |
|---|:---:|---:|
| audit the repo before we open-source it | yes | 3/3 |
| is this repo ready for the release PR? grade it against AGENTS.md | yes | 3/3 |
| run a release-readiness audit and render the report | yes | 3/3 |
| review my last commit | no | 0/3 |
| add a simulator for my robot | no | 0/3 |
| run the CPU test suite | no | 0/3 |

## Limitations

- The queries were written together with the description, so they may favor it; a held-out set written by someone else is the stronger test.
- Three runs per query resolve large differences only; one run more or less moves a query across the threshold.
- Codex was not measured.

## Reproduce

```bash
uv run python tools/skill_trigger_eval.py .agents/contributor-skills/isaac-rlpt-audit-repo
```
