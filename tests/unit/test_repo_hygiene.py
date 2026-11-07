# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository hygiene gates: naming, secrets, imports, configs, packaging.

These tests scan the working tree (tracked + untracked-unignored files) and
pin the public-release requirements:

* ZERO occurrences of the legacy project codename anywhere in the repo (the
  needle is assembled at runtime so this file itself can never contain it),
* no credential-shaped strings (HF / GitLab / W&B / AWS token prefixes),
* ``import isaac_rl_posttraining`` succeeds without the optional policy
  base models' dependencies installed (gr00t / pi0 stacks are import-guarded),
* every Hydra config referenced from configs or launchers actually exists,
* every package directory is a real package (has ``__init__.py``),
* the import root is ``isaac_rl_posttraining`` and nothing references the
  old bare package root (rename regression),
* no module under the package reads tuning values from ``os.environ``: all
  tuning flows through configs/constructor args; only a pinned allowlist of
  process-boundary variables may be read.
"""

import os
import re
import subprocess
import sys
import textwrap

import pytest
from conftest import PACKAGE_NAME, REPO_ROOT

PACKAGE_DIR = os.path.join(REPO_ROOT, PACKAGE_NAME)
CONFIGS_DIR = os.path.join(PACKAGE_DIR, "configs")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

# Needles are assembled at runtime so this test file never contains them.
LEGACY_CODENAME = "robo" + "meter"
BARE_PACKAGE_ROOT = "isaac" + "_rl"

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
    "wandb",
    ".idea",
    ".vscode",
}
_BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
    ".pt",
    ".pth",
    ".safetensors",
    ".npz",
    ".npy",
    ".so",
    ".pyc",
    ".pdf",
    ".ico",
    ".woff",
    ".woff2",
}


def _repo_files():
    """All tracked + untracked-unignored files (falls back to a filesystem walk)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
            check=False,
        )
        if out.returncode == 0:
            files = [
                os.path.join(REPO_ROOT, line)
                for line in out.stdout.splitlines()
                if line.strip()
            ]
            return [f for f in files if os.path.isfile(f)]
    except (OSError, subprocess.TimeoutExpired):
        pass
    files = []
    for root, dirs, names in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        files.extend(os.path.join(root, n) for n in names)
    return files


def _read_text(path):
    if os.path.splitext(path)[1].lower() in _BINARY_EXTS:
        return None
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None  # binary
    return data.decode("utf-8", errors="replace")


def _package_python_files():
    return [
        f
        for f in _repo_files()
        if f.endswith(".py")
        and os.path.normpath(f).startswith(os.path.normpath(PACKAGE_DIR) + os.sep)
    ]


def test_core_does_not_import_simulator_packages():
    """Simulator integrations may depend on core contracts, never the reverse."""
    offenders = []
    import_pattern = re.compile(
        r"^\s*(?:from|import)\s+simulators(?:\.|\s|$)", re.MULTILINE
    )
    for path in _package_python_files():
        text = _read_text(path) or ""
        if import_pattern.search(text):
            offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, "core modules import simulator packages:\n  " + "\n  ".join(
        offenders
    )


def test_robolab_dependency_is_owned_by_its_simulator_package():
    """Only the RoboLab integration may import the external RoboLab stack."""
    integration_dir = os.path.join(REPO_ROOT, "simulators", "robolab") + os.sep
    import_pattern = re.compile(
        r"^\s*(?:from|import)\s+robolab(?:\.|\s|$)", re.MULTILINE
    )
    offenders = []
    for path in _repo_files():
        if not path.endswith(".py") or path.startswith(integration_dir):
            continue
        text = _read_text(path) or ""
        if import_pattern.search(text):
            offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, (
        "RoboLab imports outside its simulator package:\n  " + "\n  ".join(offenders)
    )


def test_no_legacy_codename_anywhere():
    """The legacy project codename must not appear anywhere in the repository."""
    needle = LEGACY_CODENAME.lower()
    offenders = []
    for path in _repo_files():
        text = _read_text(path)
        if text is None:
            continue
        if needle in text.lower():
            lines = [
                i + 1
                for i, line in enumerate(text.splitlines())
                if needle in line.lower()
            ]
            offenders.append(f"{os.path.relpath(path, REPO_ROOT)}: lines {lines[:5]}")
    assert not offenders, (
        "legacy codename found (must be zero occurrences, reward-model references "
        "are spelled 'reward_model'):\n  " + "\n  ".join(offenders)
    )


def test_no_secret_patterns():
    """No credential-shaped strings anywhere (HF, GitLab PAT, W&B, AWS key ids)."""
    patterns = {
        "huggingface token": re.compile("hf" + "_" + r"[A-Za-z0-9]{20}"),
        "gitlab PAT": re.compile("glpat" + "-" + r"[A-Za-z0-9_\-]{10}"),
        "GitHub token": re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{30}|github_pat_[A-Za-z0-9_]{30})"
        ),
        "wandb key": re.compile("wandb" + "_v1_"),
        "aws access key id": re.compile("AKI" + "A" + r"[0-9A-Z]{16}"),
    }
    offenders = []
    for path in _repo_files():
        text = _read_text(path)
        if text is None:
            continue
        for label, rx in patterns.items():
            m = rx.search(text)
            if m:
                offenders.append(
                    f"{os.path.relpath(path, REPO_ROOT)}: {label} ({m.group(0)[:12]}...)"
                )
    assert not offenders, "credential-shaped strings found:\n  " + "\n  ".join(
        offenders
    )


def test_import_isaac_rl_without_optional_deps():
    """The package imports cleanly when optional stacks are absent.

    Covers the gr00t / pi0 model stacks, the DreamZero client extra
    (websockets + msgpack), the LIBERO simulator stack (libero / robosuite
    / mujoco), the ZeroMQ transport extra (pyzmq + msgpack-numpy), and the
    RLinf infrastructure extra (rlinf + ray) — every optional integration
    must be import-guarded.
    """
    blocked = (
        "gr00t",
        "openpi",
        "openpi_client",
        "lerobot",
        "sentence_transformers",
        "websockets",
        "msgpack",
        "msgpack_numpy",
        "zmq",
        "rlinf",
        "ray",
        "libero",
        "robosuite",
        "mujoco",
    )
    code = textwrap.dedent(
        f"""
        import sys

        class _Blocker:
            blocked = {blocked!r}

            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in self.blocked):
                    raise ImportError(f"optional dependency {{name!r}} blocked for hygiene test")
                return None

        sys.meta_path.insert(0, _Blocker())
        for mod in list(sys.modules):
            if any(mod == b or mod.startswith(b + ".") for b in _Blocker.blocked):
                del sys.modules[mod]

        import {PACKAGE_NAME}
        import {PACKAGE_NAME}.policies
        # The DreamZero package must import without its client extra: only
        # sending/receiving frames needs websockets/msgpack.
        import {PACKAGE_NAME}.policies.dreamzero
        # The transport seam must import without pyzmq (the ZeroMQ transport
        # imports its dependencies lazily, at construction).
        import {PACKAGE_NAME}.envs.zmq_transport
        # The RLinf compat package must import without rlinf/ray.
        import {PACKAGE_NAME}.compat.rlinf
        # The native runtime and its multi-process package import without Ray.
        import {PACKAGE_NAME}.runtimes.native
        import {PACKAGE_NAME}.runtimes.native_ray.rollout
        import {PACKAGE_NAME}.runtimes.native_ray.serving
        import {PACKAGE_NAME}.runtimes.native_ray.learner
        print("IMPORT_OK")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0 and "IMPORT_OK" in result.stdout, (
        f"`import {PACKAGE_NAME}` must succeed without optional base model deps.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _iter_default_entries(defaults):
    """(group, option) pairs of a defaults list; ``override /group`` and ``@package`` markers stripped."""
    for entry in defaults:
        if isinstance(entry, str):
            yield None, entry.split("@")[0]
        elif isinstance(entry, dict):
            for key, value in entry.items():
                yield str(key).removeprefix("override").strip().lstrip("/"), value


def _hydra_searchpath_dirs(doc):
    """Filesystem dirs for a config's ``hydra.searchpath`` ``pkg://`` entries.

    Co-located config groups (simulators/<sim>/configs/) reach the launch
    config through a ``pkg://`` search path, so a ``defaults`` entry may
    resolve there rather than under the config's own directory. Mirror how
    Hydra resolves them: ``pkg://a.b.configs`` -> ``a/b/configs`` in the repo.
    """
    hydra_node = doc.get("hydra")
    if not isinstance(hydra_node, dict):
        return []
    dirs = []
    for entry in hydra_node.get("searchpath") or []:
        if isinstance(entry, str) and entry.startswith("pkg://"):
            dirs.append(os.path.join(REPO_ROOT, *entry[len("pkg://") :].split(".")))
    return dirs


def _in_config_store(group: str, name: str) -> bool:
    """Whether ``register_configs()`` stores a schema node ``name`` under ``group`` (e.g. base_<name>)."""
    from hydra.core.config_store import ConfigStore

    from isaac_rl_posttraining.configs import register_configs

    register_configs()
    node = ConfigStore.instance().repo
    for part in [p for p in group.split("/") if p and p != "."]:
        node = node.get(part)
        if not isinstance(node, dict):
            return False
    return f"{name}.yaml" in node


def _load_yaml_configs(yaml, paths):
    """``({path: parsed document}, [unparseable-file messages])`` for the given YAML files."""
    docs, problems = {}, []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            try:
                docs[path] = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                problems.append(
                    f"{os.path.relpath(path, REPO_ROOT)}: unparseable YAML ({exc})"
                )
    return docs, problems


def _dangling_defaults(path, doc, searchpath_dirs):
    """The ``defaults`` entries of one config that resolve to no file or stored schema."""
    problems = []
    for group, name in _iter_default_entries(doc.get("defaults", []) or []):
        if name in (None, "_self_") or not isinstance(name, str):
            continue
        subgroup = group.split("@")[0] if group else ""
        # The entry may live under the config's own group dir, under the configs
        # root (absolute /group references), or under any co-located group
        # reached through the Hydra search path.
        group_dirs = [
            os.path.join(os.path.dirname(path), subgroup),
            os.path.join(CONFIGS_DIR, subgroup),
        ]
        group_dirs += [os.path.join(sp, subgroup) for sp in searchpath_dirs]
        candidates = [
            os.path.join(d, name + ext) for d in group_dirs for ext in (".yaml", ".yml")
        ]
        own_group = os.path.relpath(os.path.dirname(path), CONFIGS_DIR)
        if not any(os.path.isfile(c) for c in candidates) and not _in_config_store(
            subgroup or own_group, name
        ):
            problems.append(
                f"{os.path.relpath(path, REPO_ROOT)}: defaults entry "
                f"{group + ': ' if group else ''}{name} has no config file or stored schema"
            )
    return problems


def _dangling_launcher_config_names():
    """``--config-name`` references in scripts/ that name no config file."""
    problems = []
    if not os.path.isdir(SCRIPTS_DIR):
        return problems
    for root, dirs, names in os.walk(SCRIPTS_DIR):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for n in names:
            if not n.endswith((".sh", ".py")):
                continue
            text = _read_text(os.path.join(root, n)) or ""
            for m in re.finditer(r"--config-name[= ]([\w./-]+)", text):
                name = m.group(1).strip().removesuffix(".yaml")
                candidates = [
                    os.path.join(CONFIGS_DIR, name + ext) for ext in (".yaml", ".yml")
                ]
                if not any(os.path.isfile(c) for c in candidates):
                    problems.append(
                        f"scripts/{n}: --config-name={name} has no config file"
                    )
    return problems


def test_all_referenced_configs_exist():
    """Hydra defaults entries and launcher --config-name references resolve to files or stored schemas."""
    yaml = pytest.importorskip("yaml")
    assert os.path.isdir(CONFIGS_DIR), f"configs/ directory missing at {CONFIGS_DIR}"
    yaml_files = []
    for root, dirs, names in os.walk(CONFIGS_DIR):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        yaml_files.extend(
            os.path.join(root, n) for n in names if n.endswith((".yaml", ".yml"))
        )
    assert yaml_files, f"no config files shipped under {CONFIGS_DIR}"

    docs, missing = _load_yaml_configs(yaml, yaml_files)
    # Hydra's search path is process-wide, so a nested defaults list may resolve a
    # group option through the root config's searchpath.
    searchpath_dirs = [
        directory
        for doc in docs.values()
        if isinstance(doc, dict)
        for directory in _hydra_searchpath_dirs(doc)
    ]
    for path, doc in docs.items():
        if isinstance(doc, dict):
            missing.extend(_dangling_defaults(path, doc, searchpath_dirs))
    missing.extend(_dangling_launcher_config_names())

    assert not missing, "dangling config references:\n  " + "\n  ".join(missing)


def test_every_package_dir_has_init():
    """Every directory under the package that holds python code is a real package."""
    missing = []
    for root, dirs, names in os.walk(PACKAGE_DIR):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        has_py = any(n.endswith(".py") for n in names)
        has_pkg_child = any(
            os.path.isfile(os.path.join(root, d, "__init__.py")) for d in dirs
        )
        if (has_py or has_pkg_child) and "__init__.py" not in names:
            missing.append(os.path.relpath(root, REPO_ROOT))
    assert not missing, "package directories missing __init__.py:\n  " + "\n  ".join(
        missing
    )


def _pytest_module_name(path):
    """The name pytest's default import mode gives ``path``, and the directory it imports it from.

    The name is the file's dotted path below the nearest directory without an ``__init__.py``.
    """
    parts = [os.path.splitext(os.path.basename(path))[0]]
    directory = os.path.dirname(path)
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        parts.insert(0, os.path.basename(directory))
        directory = os.path.dirname(directory)
    return ".".join(parts), directory


def test_one_pytest_session_imports_every_test_module():
    """Test modules import under distinct names that shadow no repository directory.

    A test package below a plain directory becomes a top-level package; named like a
    repository directory (``tests``), it fails to import next to that directory's tests.
    """
    roots = set(os.listdir(REPO_ROOT))
    seen = {}
    problems = []
    for path in sorted(_repo_files()):
        name = os.path.basename(path)
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        module, base = _pytest_module_name(path)
        relative = os.path.relpath(path, REPO_ROOT)
        top = module.split(".")[0]
        if (
            "." in module
            and top in roots
            and os.path.normpath(base) != os.path.normpath(REPO_ROOT)
        ):
            problems.append(
                f"{relative} imports as {module!r}, shadowing the repository's {top}/"
            )
        elif module in seen:
            problems.append(f"{relative} and {seen[module]} both import as {module!r}")
        seen.setdefault(module, relative)
    assert not problems, "\n  ".join(
        ["test modules that cannot share one pytest session:", *problems]
    )


def test_no_bare_package_root_references():
    """Nothing may reference the old bare import root (word-boundary match), and its directory is gone."""
    assert not os.path.exists(os.path.join(REPO_ROOT, BARE_PACKAGE_ROOT)), (
        f"stale pre-rename package directory {BARE_PACKAGE_ROOT}/ must not exist"
    )
    rx = re.compile(r"\b" + re.escape(BARE_PACKAGE_ROOT) + r"\b")
    offenders = []
    for path in _repo_files():
        text = _read_text(path)
        if text is None:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                offenders.append(
                    f"{os.path.relpath(path, REPO_ROOT)}:{i}: {line.strip()[:100]}"
                )
    assert not offenders, (
        f"references to the old bare package root ({BARE_PACKAGE_ROOT}) remain:\n  "
        + "\n  ".join(offenders[:20])
    )


#: Environment variables package modules may READ. Everything else is tuning
#: and must flow through configs / constructor arguments. Launcher-only
#: variables (ISAACLAB_ARENA_PATH, POSTTRAIN_BASE_CKPT, POSTTRAIN_DATA_ROOT, ...) live
#: at the scripts/ process boundary, not inside the package.
ALLOWED_PACKAGE_ENV_READS = {
    "HF_TOKEN",
    "WANDB_API_KEY",
    "ISAAC_RL_POSTTRAINING_DRYRUN",
    # Process-boundary inputs of the run builders (runners/): the run-output root
    # and the simulator server addresses a launcher exports.
    "ISAAC_RL_POSTTRAINING_OUTPUT_DIR",
    "ARENA_SERVER_URL",
    "ARENA_EVAL_SERVER_URL",
    "ARENA_SIM_DEVICE",
    "TRAIN_SIM_DEVICE",
    "EVAL_SIM_DEVICE",
    # External LIBERO package's own config-location variable; read only to
    # fail fast instead of blocking on LIBERO's interactive first-run prompt
    # (no tuning flows through it).
    "LIBERO_CONFIG_PATH",
    # W&B's own run-file directory, honoured when logging.wandb_log_dir_base is unset.
    "WANDB_DIR",
    # The native Ray runtime maps plan GPUs onto the devices the driver sees, and leaves
    # Ray's own uv-run switch as the user set it.
    "CUDA_VISIBLE_DEVICES",
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
}

_ENV_READ_PATTERNS = (
    re.compile(r"os\.environ\.get\(\s*['\"](\w+)['\"]"),
    re.compile(r"os\.getenv\(\s*['\"](\w+)['\"]"),
    re.compile(r"['\"](\w+)['\"]\s+(?:not\s+)?in\s+os\.environ"),
)
_ENV_SUBSCRIPT = re.compile(r"os\.environ\[\s*['\"](\w+)['\"]\s*\]")
_ENV_WRITE = re.compile(
    r"os\.environ(\[\s*['\"]\w+['\"]\s*\]\s*=[^=]|\.setdefault\(|\.pop\(|\.update\()"
)


def test_no_retired_env_var_knobs_in_package():
    """Retired environment-variable knobs must not appear anywhere in package code.

    Multitask stats selection, the decoded action horizon and the latent
    injection knobs moved to explicit config; their env-var reads were purged
    from the GR00T base model and must never return (a stale value silently swaps
    single-task for multi-task normalization, or re-tunes the steering).
    """
    retired = (
        "GR00T_MULTITASK_DIRS",
        "GR00T_TASK_DIR",
        "GR00T_ACTION_HORIZON",
        "DSRL_LATENT_",
    )
    offenders = []
    for path in _package_python_files():
        text = _read_text(path) or ""
        for i, line in enumerate(text.splitlines(), start=1):
            for name in retired:
                if name in line:
                    offenders.append(
                        f"{os.path.relpath(path, REPO_ROOT)}:{i}: {line.strip()[:90]}"
                    )
    assert not offenders, (
        "retired multitask env vars referenced in package code:\n  "
        + "\n  ".join(offenders)
    )


def _env_access_problem(line):
    """What is wrong with one line's ``os.environ`` access, or ``None`` if nothing is."""
    if "os.environ" not in line and "os.getenv" not in line:
        return None
    names = set()
    for rx in _ENV_READ_PATTERNS:
        names.update(rx.findall(line))
    for name in _ENV_SUBSCRIPT.findall(line):
        if not re.search(r"os\.environ\[[^]]*\]\s*=[^=]", line):
            names.add(name)  # subscript read (assignment is a write)
    disallowed = {n for n in names if n not in ALLOWED_PACKAGE_ENV_READS}
    if disallowed:
        return f"reads {sorted(disallowed)}"
    if (
        not names
        and not _ENV_WRITE.search(line)
        and "environ.copy" not in line
        and "dict(os.environ)" not in line
    ):
        return f"unparseable os.environ access ({line.strip()[:90]})"
    return None


def test_no_env_var_tuning_in_package():
    """Package modules read no tuning values from os.environ (pinned allowlist only)."""
    files = _package_python_files()
    assert files, "no package python modules found"
    violations = []
    for path in files:
        text = _read_text(path) or ""
        for i, line in enumerate(text.splitlines(), start=1):
            problem = _env_access_problem(line)
            if problem:
                violations.append(f"{os.path.relpath(path, REPO_ROOT)}:{i}: {problem}")
    assert not violations, (
        "package modules must not read tuning values from os.environ "
        f"(allowlist: {sorted(ALLOWED_PACKAGE_ENV_READS)}):\n  "
        + "\n  ".join(violations)
    )


#: A module longer than this is hard to hold in one's head; split it along its responsibilities.
MODULE_LINE_BUDGET = 800
#: The leading comment block (license header and any upstream notice) and the blank line after it, not counted.
_LICENSE_HEADER = re.compile(r"(?:#.*\n)+\n?")
#: Modules already over budget when the rule landed, with the line count each may not exceed. They shrink
#: (or disappear) over time; a deleted module simply drops off the list.
MODULE_LINE_CEILINGS = {
    "simulators/g1_arena/env_server.py": 1504,
    "isaac_rl_posttraining/algorithms/sac/sac.py": 1470,
    "simulators/isaaclab/env.py": 1171,
    "isaac_rl_posttraining/policies/gr00t/data.py": 992,
    "isaac_rl_posttraining/policies/gr00t/policy.py": 849,
}


def test_module_size_budget():
    """No module grows past the line budget; the ones already over it may only shrink."""
    over = []
    for top in (
        PACKAGE_DIR,
        os.path.join(REPO_ROOT, "simulators"),
    ):
        for root, dirs, names in os.walk(top):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and d != "tests"]
            for name in names:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                relative = os.path.relpath(path, REPO_ROOT)
                with open(path, encoding="utf-8") as stream:
                    text = stream.read()
                header = _LICENSE_HEADER.match(text)
                # what `wc -l` reports, less the license header
                lines = text.count("\n") - (header.group().count("\n") if header else 0)
                ceiling = MODULE_LINE_CEILINGS.get(relative, MODULE_LINE_BUDGET)
                if lines > ceiling:
                    over.append(f"{relative}: {lines} lines (limit {ceiling})")
    assert not over, (
        "modules over the size budget (split them, or lower their ceiling in MODULE_LINE_CEILINGS only "
        "when they shrink):\n  " + "\n  ".join(sorted(over))
    )


def test_no_python311_only_exception_apis():
    """The package supports Python 3.10: no ``sys.exception()``, and notes go through ``utils.compat.add_note``."""
    helper = os.path.join(PACKAGE_DIR, "utils", "compat.py")
    offenders = []
    for path in _repo_files():
        if not path.endswith(".py") or os.path.normpath(path) == os.path.normpath(
            helper
        ):
            continue
        rel = os.path.relpath(path, REPO_ROOT)
        if rel.startswith("tests" + os.sep) or (os.sep + "tests" + os.sep) in rel:
            continue
        text = _read_text(path)
        for lineno, line in enumerate(text.splitlines(), 1):
            if "sys.exception(" in line or re.search(r"\b\w+\.add_note\(", line):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:80]}")
    assert not offenders, (
        "Python 3.11-only exception APIs (use sys.exc_info()[1] and utils.compat.add_note):\n"
        + "\n".join(offenders)
    )
