# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the Isaac RL Post-Training documentation site.

The site is markdown end to end: ``pages/`` holds the user-facing pages and the markdown
files next to this one are the developer reference, both rendered unchanged by MyST.
``docs/README.md`` has the build commands and the rule for adding a page.
"""

from __future__ import annotations

import importlib.util
import json
import logging as _stdlib_logging
import os
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent
PAGES_DIR = DOCS_DIR / "pages"

# autodoc reads the checkout, so the site builds whether or not the package is installed.
sys.path.insert(0, str(REPO_ROOT))

# Repository browser, used for the header icon and for links that leave docs/ (source
# files, example workflows). Empty unless DOCS_REPO_URL says otherwise, because a
# published page must not carry a link that only resolves inside one network: set it in
# CI once the repository has a URL its readers can open. _repo_browser_url knows the
# GitHub path shape and rejects any other host.
REPO_URL = os.environ.get("DOCS_REPO_URL", "").rstrip("/")
REPO_BRANCH = os.environ.get("DOCS_REPO_BRANCH", "main")

# Where this site is published, and which entry of the version list it is. Same rule as
# above: empty unless DOCS_BASE_URL says otherwise, and with nowhere to switch to the
# version control does not render at all.
DOCS_BASE_URL = os.environ.get("DOCS_BASE_URL", "").rstrip("/")
DOCS_VERSION = os.environ.get("DOCS_VERSION", "latest")
SWITCHER_FILE = DOCS_DIR / "_static" / "switcher.json"

# -- Project ----------------------------------------------------------------------------

project = "Isaac RL Post-Training"
author = "NVIDIA Corporation"
with (REPO_ROOT / "pyproject.toml").open("rb") as _pyproject:
    release = tomllib.load(_pyproject)["project"]["version"]
version = release

# -- Extensions -------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# Copy buttons, cards and tab sets, rendered mermaid diagrams. All three ship in the
# `docs` extra; a build without them still succeeds, with plainer output.
extensions += [
    name
    for name in ("sphinx_copybutton", "sphinx_design", "sphinxcontrib.mermaid")
    if importlib.util.find_spec(name) is not None
]

templates_path = ["_templates"]
# base_models.md, dsrl.md and ppo.md stay in the repository — the root README, the
# workflow READMEs and two launcher scripts reference them — but they are kept off
# the site, where pages/installation.md and pages/algorithms/ cover the same ground
# for the reader this site is written for.
exclude_patterns = [
    "_build",
    "README.md",
    "base_models.md",
    "dsrl.md",
    "ppo.md",
    "**/.ipynb_checkpoints",
    ".DS_Store",
]
# A docs/<name>/ folder with its own index.html is a project page, a self-contained static site
# the docs workflow copies into the built site at /<name>/; it is not documentation source.
exclude_patterns += sorted(p.name for p in DOCS_DIR.iterdir() if p.is_dir() and (p / "index.html").is_file())
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
suppress_warnings: list[str] = []

# -- Markdown (MyST) --------------------------------------------------------------------

myst_enable_extensions = [
    "attrs_inline",  # {.class} on inline markup
    "colon_fence",  # ::: directives, which survive a markdown preview
    "deflist",
    "dollarmath",  # $x$ and $$x$$
    "fieldlist",
    "html_image",  # <img> with width/alt
    "substitution",
    "tasklist",
]
# Heading anchors through level 3, so `[text](other_page.md#a-heading)` resolves.
myst_heading_anchors = 3

if "sphinxcontrib.mermaid" in extensions:
    # ```mermaid fences render as diagrams instead of unhighlighted code.
    myst_fence_as_directive = ["mermaid"]
else:
    suppress_warnings.append("misc.highlighting_failure")

# -- HTML -------------------------------------------------------------------------------

# Theme: nvidia-sphinx-theme, the NVIDIA Isaac documentation shape (the whole navigation
# in the left sidebar, plus search). Swapping it is one line: "pydata_sphinx_theme" keeps
# the options below, since the NVIDIA theme builds on pydata; "furo" needs them dropped.
html_theme = "nvidia_sphinx_theme"
if importlib.util.find_spec("nvidia_sphinx_theme") is None:
    # Fallback for an environment without it. nvidia-sphinx-theme builds on pydata, so
    # the options below still apply there.
    _has_pydata = importlib.util.find_spec("pydata_sphinx_theme") is not None
    html_theme = "pydata_sphinx_theme" if _has_pydata else "alabaster"

html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%Y-%m-%d"
html_copy_source = False

html_theme_options = {
    # The whole tree stays visible in the left sidebar: sections do not fold away as you
    # move between pages, so every page is one click from any other.
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "show_toc_level": 2,
}
if REPO_URL:
    html_theme_options["icon_links"] = [
        {
            "name": "Repository",
            "url": REPO_URL,
            "icon": "fa-solid fa-code-branch",
            "type": "fontawesome",
        },
    ]

# -- Version switcher ---------------------------------------------------------------------

# _static/switcher.json lists the published versions, newest first; a release is one entry.
# An entry's url may be relative to the site ("" for the site root, "v0.1.0/" for a
# release), and the build resolves it against DOCS_BASE_URL.


def _switcher_versions() -> list[dict[str, str]]:
    """The version list, with relative urls resolved against DOCS_BASE_URL."""
    entries = json.loads(SWITCHER_FILE.read_text())
    malformed = not isinstance(entries, list) or any(
        not isinstance(entry, dict) or "version" not in entry or "url" not in entry
        for entry in entries
    )
    if malformed:
        raise ValueError(
            f"{SWITCHER_FILE} must be a list of objects, each with a 'version' and a 'url'"
        )
    return [
        {**entry, "url": urljoin(f"{DOCS_BASE_URL}/", entry["url"])}
        for entry in entries
    ]


# Read at config time so a malformed list fails every build, not only a published one.
_versions = _switcher_versions() if SWITCHER_FILE.is_file() else []
if _versions and DOCS_BASE_URL:
    html_theme_options["switcher"] = {
        "json_url": f"{DOCS_BASE_URL}/_static/switcher.json",
        "version_match": DOCS_VERSION,
    }
    # The list is right there in the repository; no build should hang fetching it back.
    html_theme_options["check_switcher"] = False


def _publish_switcher(app: Sphinx, exception: Exception | None) -> None:
    """Write the resolved version list into the built site, over the copied source."""
    if (
        exception is not None
        or app.builder.format != "html"
        or not (DOCS_BASE_URL and _versions)
    ):
        return
    published = Path(app.builder.outdir) / "_static" / "switcher.json"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(json.dumps(_versions, indent=2) + "\n")


# -- API documentation ------------------------------------------------------------------

# autodoc directives work on a plain CPU runner: the GPU, simulator and model stacks are
# mocked, so importing a module for its docstrings never pulls one of them in.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "decord",
    "cv2",
    "wandb",
    "zmq",
    "msgpack",
    "msgpack_numpy",
    "websockets",
    "ray",
    "rlinf",
    "gr00t",
    "groot",
    "openpi_client",
    "isaaclab",
    "isaacsim",
    "omni",
    "libero",
    "robosuite",
    "mujoco",
]
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Link checking (`make linkcheck`, never part of the default build) -------------------

# Nothing is exempt: a link a reader cannot open is the finding this check exists for.
linkcheck_timeout = 20
linkcheck_retries = 2
linkcheck_anchors = False

# -- Results tables -----------------------------------------------------------------------

# A page asks for a workflow's measured results with `<!-- results: <workflow> -->` alone
# on a line. The table is rendered at build time from that workflow's results/results.json
# by the same renderer its README uses, so a page cannot hold a number the workflow no
# longer reports: refreshing results means editing one JSON file. A missing or malformed
# file fails the build rather than rendering nothing.
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"
RESULTS_PLACEHOLDER = re.compile(
    r"^[ \t]*<!--[ \t]*results:[ \t]*(?P<workflow>[\w.-]+)[ \t]*-->[ \t]*$",
    re.MULTILINE,
)


def _results_path(workflow: str) -> Path:
    return WORKFLOWS_DIR / workflow / "results" / "results.json"


def _without_intervals(table: str) -> str:
    """Drop the score intervals this site does not publish.

    A result here is a rate and the episode count behind it. Both substitutions stop
    matching once the renderer drops intervals, so this then becomes a no-op.
    """
    table = re.sub(r" \[\d+(?:\.\d+)?, ?\d+(?:\.\d+)?\]", "", table)
    return table.replace(
        "Success rates with 95% intervals in percentage points; n is the episode count.",
        "Success rates; n is the episode count.",
    )


def _drop_columns(table: str, drop: set[str]) -> str:
    """Remove named columns from a markdown table, by header.

    Keyed on the header text rather than a position, so a renderer that gains or
    reorders a column does not silently delete the wrong one.
    """
    lines = table.split("\n")
    header = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), None)
    if header is None:
        return table
    names = [c.strip() for c in lines[header].strip().split("|")[1:-1]]
    keep = [i for i, name in enumerate(names) if name not in drop]
    for i, line in enumerate(lines[header:], start=header):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = stripped.split("|")[1:-1]
        if len(cells) == len(names):
            lines[i] = "|" + "|".join(cells[j] for j in keep) + "|"
    return "\n".join(lines)


def _by_method(table: str) -> str:
    """One table per method, under its own heading.

    A single table of every task crossed with every method is a wall of rows that a
    reader has to filter by eye. Split by method and the Method column becomes the
    heading, so each table is just tasks against their numbers. The evaluator caption
    is printed once, after the last table.
    """
    lines = table.split("\n")
    header = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), None)
    if header is None:
        return table
    names = [c.strip() for c in lines[header].strip().split("|")[1:-1]]
    if "Method" not in names:
        return table
    method_at = names.index("Method")

    rows, caption = [], []
    for line in lines[header + 2 :]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            rows.append(stripped)
        elif stripped:
            caption.append(line)

    grouped: dict[str, list[str]] = {}
    for row in rows:
        cells = row.split("|")[1:-1]
        if len(cells) != len(names):
            continue
        grouped.setdefault(cells[method_at].strip(), []).append(row)
    if len(grouped) < 2:
        return table

    out = []
    for method, method_rows in grouped.items():
        block = "\n".join([lines[header], lines[header + 1], *method_rows])
        # A bold label rather than a heading: the placeholder sits at different
        # depths on different pages, and a fixed heading level breaks the hierarchy.
        out.append(f"**{method}**\n\n" + _drop_columns(block, {"Method"}))
    return "\n\n".join(out) + "\n\n" + "\n".join(caption)


def _results_markdown(workflow: str) -> str:
    """One workflow's results, rendered the way its README renders them."""
    from example_workflows._common.results_table import load_results, render

    # load_results validates the file and names the offending field when it cannot.
    results = load_results(_results_path(workflow))
    if not results.rows:
        return f"_No results published for `{workflow}` yet._"
    return _by_method(_drop_columns(_without_intervals(render(results)), {"Note"}))


def _insert_results_tables(app: Sphinx, docname: str, source: list[str]) -> None:
    """Replace every results placeholder in a page with that workflow's table."""

    def substitute(match: re.Match[str]) -> str:
        workflow = match["workflow"]
        path = _results_path(workflow)
        if not path.is_file():
            raise FileNotFoundError(
                f"{docname}: no results to show for '{workflow}' ({path} does not exist)"
            )
        # A refreshed results.json rebuilds the pages that show it.
        app.env.note_dependency(str(path))
        return _results_markdown(workflow)

    source[0] = RESULTS_PLACEHOLDER.sub(substitute, source[0])


# -- Repo-relative links ----------------------------------------------------------------


def _repo_browser_url(path: Path, anchor: str) -> str:
    """Repository browser URL for a file or directory in the checkout."""
    kind = "tree" if path.is_dir() else "blob"
    relative = path.relative_to(REPO_ROOT).as_posix()
    host = urlparse(REPO_URL).hostname or ""
    if host != "github.com":
        raise ValueError(
            f"DOCS_REPO_URL host {host!r} is not github.com; "
            "add its path shape to _repo_browser_url"
        )
    return f"{REPO_URL}/{kind}/{REPO_BRANCH}/{relative}" + (f"#{anchor}" if anchor else "")


def _existing_repo_path(candidate: Path) -> Path | None:
    """``candidate`` resolved, if it exists inside the checkout."""
    try:
        path = candidate.resolve()
    except OSError:
        return None
    return path if path.exists() and REPO_ROOT in path.parents else None


def _repo_path(docname: str, target: str) -> Path | None:
    """The checkout file or directory a markdown link points at, if it is outside docs/.

    MyST hands over either the path as written (relative to the page, or to docs/ when it
    starts with "/") or, for a path it took for a source file, an absolute path with the
    source suffix already stripped. Try each reading, restoring a suffix where one was
    dropped, and keep the first that exists. Targets inside docs/ return None: those are
    ordinary cross-references and Sphinx resolves them itself.
    """
    if not target or "://" in target or target.startswith(("#", "mailto:")):
        return None
    if target.startswith("/"):
        bases = [Path(target), DOCS_DIR / target.lstrip("/")]
    else:
        bases = [DOCS_DIR / PurePosixPath(docname).parent / target]
    for base in bases:
        candidates = [
            base,
            *(base.with_name(base.name + s) for s in source_suffix if base.name),
        ]
        for candidate in candidates:
            path = _existing_repo_path(candidate)
            if path is not None:
                return None if path == DOCS_DIR or DOCS_DIR in path.parents else path
    return None


def _link_outside_docs_to_repo(app: Sphinx, doctree: nodes.document) -> None:
    """Resolve links that leave docs/.

    Pages link to source files and example workflows the same way they would in a
    repository markdown file (``../../example_workflows/...``). Those paths are not part
    of the documentation build, so send them to the repository browser when one is
    configured, and otherwise keep the path as text: a dead link, a raw file offered as a
    download, and a link into a repository the reader cannot open are all worse than the
    path itself. Links inside docs/, and links to paths that do not exist, are untouched,
    so a typo still reports itself.
    """
    docname = app.env.docname
    for node in list(doctree.findall(addnodes.download_reference)) + list(
        doctree.findall(addnodes.pending_xref)
    ):
        if node.get("reftype") != "myst" or node.get("refdomain") not in (
            None,
            "",
            "doc",
        ):
            continue
        target = str(node.get("reftarget", ""))
        if target in app.env.found_docs:  # a page of this site
            continue
        target, _, anchor = target.partition("#")
        path = _repo_path(docname, target)
        if path is None:
            continue
        anchor = anchor or str(node.get("reftargetid") or "")
        text = node.children or [nodes.Text(target)]
        if REPO_URL:
            replacement = nodes.reference(
                "", "", refuri=_repo_browser_url(path, anchor), internal=False
            )
            replacement += text
        else:
            replacement = nodes.inline("", "", *text)
        node.replace_self(replacement)


# -- While the user-facing pages are being written ---------------------------------------

# index.md wires up pages/ before its first page exists. Until one lands, the missing
# nav entries and cross-links are expected, and a fresh checkout still builds clean; the
# leniency disarms itself as soon as pages/ holds any markdown.
_pages_written = PAGES_DIR.is_dir() and any(PAGES_DIR.rglob("*.md"))
if not _pages_written:
    suppress_warnings += ["toc.not_readable", "myst.xref_missing", "ref.doc"]


class _QuietEmptyGlob(_stdlib_logging.Filter):
    """Drop 'glob pattern didn't match any documents': it carries no suppressible type."""

    def filter(self, record: _stdlib_logging.LogRecord) -> bool:
        return "didn't match any documents" not in record.getMessage()


def setup(app: Sphinx) -> None:
    app.connect("source-read", _insert_results_tables)
    app.connect("doctree-read", _link_outside_docs_to_repo)
    app.connect("build-finished", _publish_switcher)
    if not _pages_written:
        _stdlib_logging.getLogger("sphinx").addFilter(_QuietEmptyGlob())
