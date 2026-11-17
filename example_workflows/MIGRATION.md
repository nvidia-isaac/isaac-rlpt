# Migrating an existing example into `example_workflows/`

An example moves in two pull requests into `main`, so nothing a colleague relies on breaks before its replacement is validated:

- **Pull request A** creates the workflow by copying the example's material; the old location keeps working.
- **Pull request B** retires the old location once the new workflow is validated.

The simulator packages themselves (`simulators/<name>/env.py`, `runtime.py`, `configs/`, their tests) are library-side integrations and stay where they are; a workflow imports them.

## Pull request A: create the workflow

1. Follow every step of [Start a new workflow](../docs/pages/workflows/template.md), with these differences:
   - Copy, do not move, the old location's scripts, configs, task lists, analysis scripts and README content.
   - Fill `results/results.json` from the numbers the old README or its reports carry.
   - The folder may go under `example_workflows/experimental/<workflow>/`.
2. Put everything internal under the workflow's `internal/` directory: cluster job templates, container files that pull from the internal registry, node provisioning and sync scripts, shared-filesystem paths, internal hostnames.
   `example_workflows/_common/release.py` lists the terms, and `tests/unit/test_example_workflows.py` fails if they appear elsewhere.
3. Add one line at the top of the old README: "This example is moving to `example_workflows/<workflow>/`; the copy there is the maintained one."
4. Open the pull request into `main` and paste the checklist below into its description.

### Acceptance checklist for pull request A

- [ ] Folder follows `_template/` (README sections in order, `smoke.sh`, `results/results.json`, `__init__.py`)
- [ ] `smoke.sh` passes in under two minutes without a GPU or external repositories
- [ ] Every recipe's resolved config is pinned in `tests/golden_config.json`
- [ ] Every internal reference is under `internal/`; the layout test passes
- [ ] README results table is rendered from `results/results.json`; every number has base, episodes, evaluator, commit, date
- [ ] Old location untouched apart from the pointer line in its README
- [ ] Lint, types and the CPU test suite green

## Pull request B: retire the old location

1. Delete the old folder except the scripts colleagues launch; replace each of those with a stub that prints the new path and executes it, kept for one release.
2. Update every reference to the old paths in `README.md`, `AGENTS.md`, `docs/`, the simulator READMEs and the CI configuration; `tests/unit/test_doc_references.py` reports any remaining references.
3. Run the verification step of [Start a new workflow](../docs/pages/workflows/template.md#steps).
