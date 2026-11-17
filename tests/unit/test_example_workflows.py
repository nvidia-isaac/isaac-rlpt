# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example_workflows layout: what every workflow folder must contain, and where internal material may live.

release-scan: exempt (the fixtures below spell out internal terms on purpose)
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import REPO_ROOT  # test scaffolding

EXAMPLES = Path(REPO_ROOT) / "example_workflows"
TEMPLATE = EXAMPLES / "_template"

sys.path.insert(0, str(REPO_ROOT))
from example_workflows._common import release, smoke
from example_workflows._common.results_table import (
    format_rate,
    load_results,
    render,
    table_matches,
)

REQUIRED_SECTIONS = (
    "## Overview",
    "## Requirements",
    "## Quickstart",
    "## Results",
    "## Method",
    "## Files",
)
STATUS_LINE = re.compile(r"^\*\*Status:\*\* ", re.MULTILINE)


def _workflows() -> list[Path]:
    return [TEMPLATE, *smoke.workflow_dirs(EXAMPLES)]


def _ids(path: Path) -> str:
    return path.relative_to(EXAMPLES).as_posix()


@pytest.mark.parametrize("workflow", _workflows(), ids=_ids)
def test_workflow_folder_has_the_required_files(workflow: Path) -> None:
    for required in ("README.md", "smoke.sh", "results/results.json", "__init__.py"):
        assert (workflow / required).exists(), f"{_ids(workflow)} lacks {required}"
    assert os.stat(workflow / "smoke.sh").st_mode & stat.S_IXUSR, (
        f"{_ids(workflow)}/smoke.sh must be executable"
    )


@pytest.mark.parametrize("workflow", _workflows(), ids=_ids)
def test_readme_follows_the_template_sections_in_order(workflow: Path) -> None:
    text = (workflow / "README.md").read_text(encoding="utf-8")
    positions = [text.find(section) for section in REQUIRED_SECTIONS]
    missing = [
        section
        for section, position in zip(REQUIRED_SECTIONS, positions)
        if position < 0
    ]
    assert not missing, f"{_ids(workflow)}/README.md lacks sections {missing}"
    assert positions == sorted(positions), (
        f"{_ids(workflow)}/README.md sections are out of template order"
    )
    assert STATUS_LINE.search(text), (
        f"{_ids(workflow)}/README.md needs a '**Status:** ...' line"
    )


@pytest.mark.parametrize("workflow", _workflows(), ids=_ids)
def test_readme_results_table_matches_results_json(workflow: Path) -> None:
    results = load_results(workflow / "results" / "results.json")
    assert table_matches(results, workflow / "README.md"), (
        f"{_ids(workflow)}: README results table is out of date; run "
        f"python -m example_workflows._common.results_table {_ids(workflow)}/results/results.json --write README.md"
    )


def test_internal_material_only_lives_under_internal_directories() -> None:
    """The whole repository, not only example_workflows/: a release strips every internal/ directory."""
    root = Path(REPO_ROOT)
    findings = release.internal_references(root)
    assert not findings, (
        "internal terms, paths into internal/ directories, or unfenced release-excluded features:\n  "
        + "\n  ".join(finding.describe(root) for finding in findings)
    )


def test_internal_scan_catches_terms_and_paths(tmp_path: Path) -> None:
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "job.yaml").write_text(
        "image: nvcr.io/team/image\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "Submit with the OSMO template.\nSee internal/job.yaml.\nA bare internal/ mention is fine.\nCosmos is fine.\n",
        encoding="utf-8",
    )
    findings = release.internal_references(tmp_path)
    assert [(finding.line, finding.label) for finding in findings] == [
        (1, "cluster scheduler name"),
        (2, "path into an internal/ directory"),
    ]
    assert release.internal_dirs(tmp_path) == [tmp_path / "internal"]


def test_internal_scan_catches_internal_links_and_names(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        # Split spellings keep the terms themselves out of the released tree.
        "git clone ssh://git@gitlab" + "-master.nvidia.com:12051/team/repo.git\n"
        "git checkout mm" + "oghani-fork\n"
        "git checkout masou" + "d/g1_factory_release\n"
        "It lives in the NVIDIA" + "-internal repository.\n"
        "See https://confluence.nvidia.com/display/X and nvbugs 123.\n"
        "Data at swift://pdx.s8k.io/AUTH_team-x/data.\n"
        "git clone --recursive https://github.com/NVlabs/G1-Factory.git\n"
        "uvx hf download masoudmoghani/G1-Factory --repo-type dataset\n"
        "See https://docs.nvidia.com/ and the internal/ mechanism.\n",
        encoding="utf-8",
    )
    findings = release.internal_references(tmp_path)
    assert [(finding.line, finding.label) for finding in findings] == [
        (1, "internal source host"),
        (2, "internal user namespace"),
        (3, "internal branch name"),
        (4, "internal-only wording"),
        (5, "internal host"),
        (6, "object-store path"),
    ]


def test_internal_scan_allows_only_public_ngc_images(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "docker run nvcr.io/nvidia/isaac-lab:3.0.0-beta2\n"
        "run `docker login nvcr.io` with an NGC API key\n"
        "image: nvcr.io/team/image\n",
        encoding="utf-8",
    )
    findings = release.internal_references(tmp_path)
    assert [(finding.line, finding.label) for finding in findings] == [
        (3, "internal container registry"),
    ]


def test_strip_internal_removes_every_internal_directory_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    copy = tmp_path / "example_workflows"
    shutil.copytree(EXAMPLES, copy, ignore=shutil.ignore_patterns("__pycache__", ".*"))
    features = release.release_features(Path(REPO_ROOT))
    if features:  # the fences in example_workflows/ name these features; their paths lie outside the copy
        listing = copy / release.RELEASE_FEATURES_FILE
        listing.parent.mkdir(parents=True)
        listing.write_text(
            "".join(
                f"[features.{name}]\nterms = {[term.pattern for term in feature.terms]!r}\n"
                for name, feature in features.items()
            ),
            encoding="utf-8",
        )
    if not release.internal_dirs(copy):
        pytest.skip("release tree: the internal/ directories are already stripped")
    spec = importlib.util.spec_from_file_location(
        "strip_internal", Path(REPO_ROOT) / "tools" / "strip_internal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--root", str(copy), "--check"]) == 0
    assert module.main(["--root", str(copy)]) == 0

    assert release.internal_dirs(copy) == []
    assert (copy / "_common" / "release.py").exists() and (
        copy / "_template" / "README.md"
    ).exists()
    assert release.internal_references(copy) == []


def test_strip_internal_removes_the_internal_only_repository_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "example_workflows").mkdir()
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    for name in release.INTERNAL_ONLY_PATHS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("internal\n", encoding="utf-8")
    (workflows / "docs.yml").write_text("public\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "strip_internal", Path(REPO_ROOT) / "tools" / "strip_internal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--root", str(tmp_path)]) == 0
    assert not any((tmp_path / name).exists() for name in release.INTERNAL_ONLY_PATHS)
    assert (workflows / "docs.yml").exists(), "only the listed files are internal"


def test_strip_internal_refuses_while_references_remain(tmp_path: Path) -> None:
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "job.yaml").write_text(
        "image: nvcr.io/team/image\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "Submit internal/job.yaml to the cluster.\n", encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location(
        "strip_internal", Path(REPO_ROOT) / "tools" / "strip_internal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--root", str(tmp_path)]) == 1
    assert (tmp_path / "internal" / "job.yaml").exists(), (
        "nothing is removed while the stripped tree would dangle"
    )


def _strip_internal():
    spec = importlib.util.spec_from_file_location(
        "strip_internal", Path(REPO_ROOT) / "tools" / "strip_internal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_tree(root: Path, readme: str) -> None:
    """A tree with one release-excluded feature, ``demo``, that owns ``demo/`` and is named by ``demo_backend``."""
    listing = root / release.RELEASE_FEATURES_FILE
    listing.parent.mkdir(parents=True)
    listing.write_text(
        '[features.demo]\npaths = ["demo"]\nterms = ["demo_backend"]\n',
        encoding="utf-8",
    )
    (root / "demo").mkdir()
    (root / "demo" / "backend.py").write_text(
        "NAME = 'demo_backend'\n", encoding="utf-8"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")


def test_strip_internal_removes_a_release_excluded_feature(tmp_path: Path) -> None:
    _feature_tree(
        tmp_path,
        "Runtimes:\n"
        "- native\n"
        "- demo_backend  <!-- release-exclude: demo -->\n"
        "<!-- release-exclude-begin: demo -->\n"
        "Set up demo_backend first.\n"
        "<!-- release-exclude-end: demo -->\n"
        "The end.\n",
    )
    strip = _strip_internal()

    assert strip.main(["--root", str(tmp_path), "--check"]) == 0
    assert (tmp_path / "demo").exists(), "--check removes nothing"
    assert strip.main(["--root", str(tmp_path)]) == 0

    assert not (tmp_path / "demo").exists()
    assert not (tmp_path / release.RELEASE_FEATURES_FILE).exists()
    assert (tmp_path / "README.md").read_text(
        encoding="utf-8"
    ) == "Runtimes:\n- native\nThe end.\n"
    assert release.internal_references(tmp_path) == []


def test_strip_internal_refuses_a_dangling_reference_to_an_excluded_feature(
    tmp_path: Path,
) -> None:
    _feature_tree(tmp_path, "Runtimes: native, demo_backend.\n")

    findings = release.internal_references(tmp_path)
    assert [(finding.line, finding.label) for finding in findings] == [
        (1, "release-excluded feature demo")
    ]
    assert _strip_internal().main(["--root", str(tmp_path)]) == 1
    assert (tmp_path / "demo").exists(), (
        "nothing is removed while the stripped tree would dangle"
    )


def test_strip_internal_refuses_a_missing_feature_path(tmp_path: Path) -> None:
    _feature_tree(tmp_path, "Runtimes: native.\n")
    shutil.rmtree(tmp_path / "demo")

    findings = release.internal_references(tmp_path)
    assert [(finding.label, finding.text) for finding in findings] == [
        ("release-excluded feature demo", "lists the missing path demo")
    ]
    assert _strip_internal().main(["--root", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("<!-- release-exclude: other -->\n", "unlisted feature 'other'"),
        ("<!-- release-exclude-begin: demo -->\n", "never closes"),
        ("<!-- release-exclude-end: demo -->\n", "not open"),
    ],
)
def test_malformed_release_fences_are_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    _feature_tree(tmp_path, text)

    findings = release.internal_references(tmp_path)
    assert [finding.label for finding in findings] == ["malformed release fence"]
    assert problem in findings[0].text


def test_relock_gives_the_workflow_lockfiles_the_root_package_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strip = _strip_internal()
    monkeypatch.setattr(
        strip, "subprocess", SimpleNamespace(run=lambda *a, **k: None)
    )  # uv lock
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n', encoding="utf-8"
    )

    def lock(extras: str) -> str:
        return (
            '[[package]]\nname = "demo"\nsource = { editable = "." }\n\n'
            f"[package.metadata]\nprovides-extras = [{extras}]\n\n"
            '[[package]]\nname = "numpy"\nversion = "2.0"\n'
        )

    (tmp_path / "uv.lock").write_text(lock('"cpu"'), encoding="utf-8")
    (tmp_path / "workflow").mkdir()
    (tmp_path / "workflow" / "uv.lock").write_text(
        lock('"cpu", "gone"'), encoding="utf-8"
    )

    strip.relock(tmp_path)

    assert (tmp_path / "workflow" / "uv.lock").read_text(encoding="utf-8") == lock(
        '"cpu"'
    )


def test_rate_renders_with_its_denominator() -> None:
    assert format_rate(173, 200) == "86.5% n=200"
    assert format_rate(0, 10) == "0.0% n=10"
    assert format_rate(10, 10) == "100.0% n=10"
    with pytest.raises(ValueError, match="episodes must be positive"):
        format_rate(0, 0)
    with pytest.raises(ValueError, match="successes must lie"):
        format_rate(11, 10)


def test_results_table_renders_base_peak_and_final(tmp_path: Path) -> None:
    results = load_results(TEMPLATE / "results" / "results.json")
    table = render(results)
    assert (
        "| ExampleTask | ExampleMethod | 50.0% n=100 | 60.0% n=100 | 60.0% n=100 | illustrative |"
        in table
    )
    assert "| Task | Method | Base | Peak | Final | Note |" in table
    assert "Measured 1970-01-01." in table

    bad = tmp_path / "results.json"
    bad.write_text(
        '{"workflow": "w", "evaluator": "e", "date": "d", "rows": [{"task": "t", "method": "m", "base": {"successes": 5, "episodes": 4}, "peak": null, "final": {"successes": 1, "episodes": 4}}]}',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match=r"rows\[0\].base: need 0 <= successes <= episodes"
    ):
        load_results(bad)


def test_seeded_results_render_one_row_per_task(tmp_path: Path) -> None:
    def row(task: str, method: str, base: int, best: int) -> dict:
        return {
            "task": task,
            "method": method,
            "base": {"successes": base, "episodes": 50},
            "peak": {"successes": best, "episodes": 50},
            "final": None,
        }

    rows = [
        row("A", "DSRL", 25, 40),
        row("A", "DSRL", 25, 30),
        row("A", "PPO", 20, 45),
        row("B", "DSRL", 10, 10),
        row("B", "DSRL", 10, 20),
    ]
    data = {"workflow": "w", "evaluator": "e", "date": "d", "rows": rows}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    table = render(load_results(path))
    assert "| Task | DSRL | PPO |" in table
    assert "| A | 50% → 70% | 40% → 90% |" in table  # (40 + 30) / 100 pooled
    assert "| B | 20% → 30% | — |" in table
    assert "re-scored on 50 held-out episodes; the 2 training seeds are pooled" in table

    data["rows"][1]["base"]["successes"] = 26
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="different bases"):
        render(load_results(path))


def test_smoke_runner_discovers_workflows_and_reports_failures(tmp_path: Path) -> None:
    for name, code in (("passing", 0), ("experimental/failing", 1), ("_skipped", 0)):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        (folder / "smoke.sh").write_text(
            f"#!/usr/bin/env bash\necho running {name}\nexit {code}\n", encoding="utf-8"
        )
    (tmp_path / "no_smoke").mkdir()
    (tmp_path / "no_smoke" / "README.md").write_text(
        "# no smoke check\n", encoding="utf-8"
    )

    names = [
        smoke.workflow_name(path, tmp_path) for path in smoke.workflow_dirs(tmp_path)
    ]
    assert names == ["no_smoke", "passing", "experimental/failing"]

    outcomes = {
        smoke.workflow_name(p, tmp_path): smoke.run_smoke(p, timeout=30, root=tmp_path)
        for p in smoke.workflow_dirs(tmp_path)
    }
    assert (
        outcomes["passing"].passed and "running passing" in outcomes["passing"].output
    )
    assert not outcomes["experimental/failing"].passed
    assert (
        not outcomes["no_smoke"].passed
        and "does not exist" in outcomes["no_smoke"].output
    )


def test_template_smoke_check_passes() -> None:
    outcome = smoke.run_smoke(TEMPLATE, timeout=300, root=EXAMPLES)
    assert outcome.passed, outcome.output[-3000:]
    assert "smoke check passed" in outcome.output


def test_smoke_runner_cli_lists_and_runs() -> None:
    listed = subprocess.run(
        [sys.executable, "-m", "example_workflows._common.smoke", "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "_template" not in listed.stdout, (
        "the template is a skeleton, not a workflow"
    )
    unknown = subprocess.run(
        [sys.executable, "-m", "example_workflows._common.smoke", "--only", "nope"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unknown.returncode == 2 and "unknown workflows" in unknown.stderr


def test_the_golden_commands_its_docstring_shows_parse():
    from example_workflows._common import golden

    commands = re.findall(
        r"python -m example_workflows\._common\.golden (.+)", golden.__doc__ or ""
    )
    assert commands
    for command in commands:
        golden.argument_parser().parse_args(command.split())
