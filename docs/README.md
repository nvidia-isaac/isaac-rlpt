# Documentation site

This folder builds the documentation site with [Sphinx](https://www.sphinx-doc.org) and [MyST](https://myst-parser.readthedocs.io), so every page is plain markdown that reads the same in a repository preview.

```
docs/pages/       the user-facing site: getting started, workflows, algorithms, guides
docs/*.md         the developer reference: how the library is put together
docs/conf.py      the build: theme, extensions, navigation rules
docs/<name>/      a project page: a self-contained static site with its own index.html, served at /<name>/
```

## Build

Run from the repository root:

```bash
uv sync --extra docs                # add --extra cpu on a machine without a GPU
uv run make -C docs html            # writes docs/_build/html/index.html
uv run make -C docs serve           # the same, then serves it on http://localhost:8000
```

The site needs neither the project nor a GPU, so it also builds without the project environment:

```bash
uv run --no-project --with sphinx --with myst-parser --with nvidia-sphinx-theme \
       --with sphinx-copybutton --with sphinx-design --with sphinxcontrib-mermaid \
       sphinx-build -b html docs docs/_build/html
```

- `make -C docs strict` turns every warning into an error, so a dead cross-reference or a page missing from the navigation fails the build. Run it before you push documentation changes.
- `make -C docs linkcheck` checks external links. It needs the network.

CI runs the strict build on every pull request (`.github/workflows/docs.yml`) and publishes it to GitHub Pages from the default branch.

## Adding a page

A markdown file dropped into one of these folders appears in the sidebar on the next build, sorted by file name:

| Folder | Section in the sidebar |
|---|---|
| `pages/workflows/` | Workflows (`index.md` is pinned first) |
| `pages/algorithms/` | Algorithms |
| `pages/extending/` | Extending |
| `docs/` (top level) | Developer reference |

The `Get started` and `Guides` sections list their pages explicitly, so a new page there needs one line in a toctree in `docs/index.md`.
A page in no toctree is a build warning ("document isn't included in any toctree").

## Page syntax

- **MyST markdown.** Directives work in either fence, ` ```{note} ` or `:::{note}`. Definition lists, field lists, task lists, `$math$` and `$$math$$`, inline attributes and raw `<img>` tags are enabled.
- **Directives.** The admonitions (`{note}`, `{warning}`, `{tip}`, ...), `{toctree}`, `{code-block}`, `{list-table}`, and from [sphinx-design](https://sphinx-design.readthedocs.io) `{grid}`, `{grid-item-card}`, `{tab-set}`, `{tab-item}` and `{dropdown}`. Every code block gets a copy button.
- **Diagrams.** A ` ```mermaid ` fence renders as a diagram; `docs/runtime-class-interactions.md` has examples.
- **Links between pages.** Use a relative link to the file: `[installation](installation.md)`, `[the data model](../../data_model.md)`. Headings down to level 3 get anchors: `[the budget](../concepts.md#decision-budget)`.
- **Links into the repository.** Write them relative to the page, as in any repository markdown file: `[the launcher](../../../example_workflows/g1_arena_posttraining/scripts/train.sh)`. A link that leaves `docs/` becomes a repository-browser link when `DOCS_REPO_URL` is set, and plain text otherwise. A link to a missing path is a warning.
- **Images.** Put them next to the page and reference them relatively: `![a rollout](images/rollout.png)`. `docs/_static/` is for site-wide assets only.
- **Results.** Do not type result numbers; name the workflow with `<!-- results: <workflow> -->` and the build renders its table (see [Results tables](#results-tables)).
- **API documentation.** `{eval-rst}` blocks with `automodule` or `autoclass` work on a CPU runner, because `conf.py` mocks the GPU, simulator and model stacks.

## Results tables

A page never types a result.
It names a workflow, alone on a line:

```
<!-- results: g1_arena_posttraining -->
```

The build replaces that line with the table rendered from `example_workflows/<workflow>/results/results.json` by `example_workflows/_common/results_table.py`, the same renderer the workflow README uses.
To refresh results, edit that JSON file.

- Every cell keeps its episode count, and the evaluator caption stays under the table.
- A workflow with empty `rows` renders "No results published for `<workflow>` yet".
- A missing `results.json`, or one that fails validation, fails the build and names the path or field.

## Conventions

- Write a result as a rate with its episode count, `86.5% n=200`, next to the same task's base number. Do not add an interval or a statement about what the difference means.
- Prefer plain words: "runs", "task and algorithm combinations".
- One sentence per line in the source, so a diff shows the sentence that changed.

## Project pages

A folder `docs/<name>/` that holds its own `index.html` is a project page, not documentation: a self-contained static site.
The Sphinx build leaves such folders alone (`conf.py` excludes them), and the GitHub workflow copies each one into the built site before the Pages deployment, so the page appears at `<documentation site>/<name>/` on the next push to the default branch and shares the site's visibility.
Keep every path inside a page relative; serve it locally with `python3 -m http.server` from its folder, since pages that load data with `fetch()` do not work from a `file://` URL.

## Publishing

`DOCS_REPO_URL` (and `DOCS_REPO_BRANCH`, default `main`) adds the header link and turns out-of-docs paths into repository links; without it the site has no repository link.
Set it as a CI variable once the repository has a URL its readers can open.
GitHub (`<repo>/blob/<branch>/<path>`) is supported; any other host fails the build.
The GitHub workflow sets both variables itself.

The GitHub workflow publishes on every push to the default branch.
Pages needs GitHub Actions as its source (Settings > Pages), where the site's visibility is also set: Pages on a private repository can serve a public site unless the organization restricts it.

### Versions

The header shows a version switcher only when `DOCS_BASE_URL` is set.
`docs/_static/switcher.json` lists the versions, newest first, one entry per release:

```json
{ "name": "0.2.0", "version": "0.2.0", "url": "v0.2.0/" }
```

A `url` is relative to `DOCS_BASE_URL` (`""` is the site root) or absolute.
`DOCS_VERSION` names the entry being built, default `latest`.

## Theme

The theme is `nvidia-sphinx-theme`, set by `html_theme` in `conf.py`; `pydata-sphinx-theme` and `furo` are drop-in alternatives.
