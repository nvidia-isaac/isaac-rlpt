# Contributing to Isaac RL Post-Training

Contributions are welcome as pull requests against `main`.
`AGENTS.md` holds the codebase map, the invariants with the test that enforces each, and the coding principles every change follows; read it first.

## Before you open a request

1. Install the environment as in `docs/pages/installation.md` (`uv sync --extra cpu` on a machine without a GPU).
2. Run the CPU test suite and every workflow's smoke check from the repository root:

   ```bash
   uv run --extra cpu pytest tests/ simulators/*/tests/ example_workflows/*/tests/ -m "not gpu and not arena and not libero and not slow"
   uv run --extra cpu python -m example_workflows._common.smoke
   ```

3. Build the documentation strictly when you change `docs/`:

   ```bash
   uv run --extra cpu --extra docs sphinx-build -b html docs docs/_build/html -W --keep-going
   ```

4. Keep each change focused, and split unrelated changes into separate requests.
   A reported number goes in its workflow's `results/results.json` with the commit and evaluation protocol that produced it.

Extending the library has a guide and an agent skill for each extension point:

| To add | Guide | Skill |
|---|---|---|
| a simulator or robot | `docs/pages/extending/add_a_simulator.md` | `skills/isaac-rlpt-add-simulator/` |
| a base model | `docs/pages/extending/add_a_base_model.md` | `skills/isaac-rlpt-add-base-model/` |
| an algorithm | `docs/pages/extending/add_an_algorithm.md` | `skills/isaac-rlpt-add-algorithm/` |
| an example workflow | `docs/pages/workflows/template.md` | none |

## Signing your work

Every commit must carry a `Signed-off-by` line certifying the Developer Certificate of Origin below.
`git commit -s` adds it from your `user.name` and `user.email`:

```
Signed-off-by: Your Name <your@email.com>
```

Requests with unsigned commits cannot be accepted.

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## License

Contributions are licensed under the Apache License 2.0 in `LICENSE`.
The agent skills under `skills/` and `.agents/contributor-skills/` are licensed under CC-BY-4.0 AND Apache-2.0: their documentation files, such as `SKILL.md`, under both licenses (`LICENSE-CC-BY-4.0` and `LICENSE`), and their code-only files under the Apache License 2.0.

## Security

Report security vulnerabilities as described in `SECURITY.md`, not in a public issue.
