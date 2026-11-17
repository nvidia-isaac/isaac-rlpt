# Isaac RL Post-Training

Isaac RL Post-Training is a library for raising the success rate of a pretrained robot policy on a chosen task with reinforcement learning in simulation.

```bash
git clone https://github.com/nvidia-isaac/isaac-rlpt.git
cd isaac-rlpt
uv sync
uv run python -m example_workflows._common.smoke   # about two minutes, no GPU needed
```

## Contents

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: pages/getting_started
:link-type: doc

The steps to a first run.
:::

:::{grid-item-card} Installation
:link: pages/installation
:link-type: doc

Every install variant: base, CPU-only, extras, base models, simulators.
:::

:::{grid-item-card} Concepts
:link: pages/concepts
:link-type: doc
:::

:::{grid-item-card} Workflows
:link: pages/workflows/index
:link-type: doc

Complete recipes per simulator: what to install, what to run, what comes out.
:::

:::{grid-item-card} Algorithms
:link: pages/algorithms/dsrl
:link-type: doc

Online (DSRL, RFS, SAC-edit, PPO) and offline (SFT, IQL): what each one trains and when to pick it.
:::

::::

The left sidebar also has the [command-line reference](pages/cli_reference.md), [troubleshooting](pages/troubleshooting.md), the extending guides ([simulator](pages/extending/add_a_simulator.md), [algorithm](pages/extending/add_an_algorithm.md), [base model](pages/extending/add_a_base_model.md)), and the developer reference.

## License

Isaac RL Post-Training is released under the [Apache 2.0 License](https://github.com/nvidia-isaac/isaac-rlpt/blob/main/LICENSE).
The agent skills under `skills/` and `.agents/contributor-skills/` are licensed under CC-BY-4.0 AND Apache-2.0 ([CC-BY-4.0](https://github.com/nvidia-isaac/isaac-rlpt/blob/main/LICENSE-CC-BY-4.0) and [Apache 2.0](https://github.com/nvidia-isaac/isaac-rlpt/blob/main/LICENSE)); code-only files in them are Apache 2.0.
Third-party code and licenses: see [THIRD_PARTY_NOTICES.md](https://github.com/nvidia-isaac/isaac-rlpt/blob/main/THIRD_PARTY_NOTICES.md).

This repository retrieves external materials that it does not distribute: the workflows' `fetch_externals.sh` scripts and `uv sync` download third-party source code and Python packages, base models and checkpoints are downloaded from Hugging Face, and the workflow Dockerfiles build images locally from third-party base images.
No container image is distributed with this repository; an image you build contains only what your build retrieves.

**NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.  Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.**

```{toctree}
:caption: Get started
:maxdepth: 1
:hidden:

pages/getting_started
pages/installation
pages/concepts
```

```{toctree}
:caption: Workflows
:maxdepth: 1
:hidden:

pages/workflows/index
pages/workflows/template
pages/workflows/g1_arena
pages/workflows/isaaclab_gr00t
pages/workflows/libero
pages/workflows/v2d_box_lift
pages/workflows/dgx_example
```

```{toctree}
:caption: Online RL algorithms
:maxdepth: 1
:hidden:

pages/algorithms/dsrl
pages/algorithms/rfs
pages/algorithms/sac_edit
pages/algorithms/ppo
```

```{toctree}
:caption: Offline RL algorithms
:maxdepth: 1
:hidden:

pages/algorithms/sft
pages/algorithms/iql
```

```{toctree}
:caption: Guides
:maxdepth: 1
:hidden:

pages/cli_reference
pages/troubleshooting
```

```{toctree}
:caption: Extending
:maxdepth: 1
:glob:
:hidden:

pages/extending/*
```

```{toctree}
:caption: Developer reference
:maxdepth: 1
:glob:
:hidden:

*
```
