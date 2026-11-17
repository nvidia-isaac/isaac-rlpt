# isaac-rlpt-add-simulator benchmark

## What is measured

Discovery: how often an agent loads this skill on its own for a prompt that needs it, and how often it loads it for a prompt that does not.
Each query in `evals/evals.json` (`trigger_evals`) runs 3 times in a scratch copy of the repository, with this skill as the only project skill; a query counts as triggered when the agent calls the Skill tool on this skill in at least 2 of the 3 runs, within 90 seconds.
The task evaluations in `evals/evals.json` (`evals`), which grade a full run against its expectations, have not been run yet.

## Results

Measured with Claude Code.

| Positive queries triggered | Positive runs | False triggers | Negative runs triggered |
|---:|---:|---:|---:|
| 7 of 7 | 21 of 21 | 0 of 7 | 0 of 21 |

| Query | Should trigger | Triggered |
|---|:---:|---:|
| how do I plug my own robot simulator into isaac rlpt so I can run DSRL on it? | yes | 3/3 |
| I wrote a gymnasium env for my arm, what do I need to write so the rollout workers can build it? | yes | 3/3 |
| dry run says: no simulator entry is registered for environment 'my_robot/reach'. what now? | yes | 3/3 |
| getting 'data specs of simulator my_robot and the policy disagree: action: emitted float32[35] x 40 steps vs accepted float32[7]' with base_model=mock | yes | 3/3 |
| my simulator has two cameras, wrist and front, how do I tell RLPT which observation keys are images? | yes | 3/3 |
| I want to drive a real robot through the env server transport instead of a simulator, how should the bridge look? | yes | 3/3 |
| add a mujoco task suite as a new simulator under simulators/ with its own config group | yes | 3/3 |
| run the G1 Factory DrillLift DSRL training on GPU 2 | no | 0/3 |
| evaluate the released PushButton checkpoint on LIBERO for 100 episodes | no | 0/3 |
| add a new RL algorithm that edits the decoded action chunk | no | 0/3 |
| wrap OpenVLA as a new base model so DSRL can steer it | no | 0/3 |
| why did my DSRL run's success rate collapse after 45k steps? | no | 0/3 |
| build the docs site locally | no | 0/3 |
| what does dsrl.noise_action_bound do? | no | 0/3 |

## Limitations

- The queries were written together with the description, so they may favor it; a held-out set written by someone else is the stronger test.
- Three runs per query resolve large differences only; one run more or less moves a query across the threshold.
- Codex was not measured.

## Reproduce

```bash
uv run python tools/skill_trigger_eval.py skills/isaac-rlpt-add-simulator
```
