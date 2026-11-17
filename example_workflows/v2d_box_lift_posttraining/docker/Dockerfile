# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The simulator image built by the "v2d" bake target (see docker-bake.hcl).
FROM v2d_base:bake
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /usr/local/bin/uv

# Every learner package is pinned and hashed; the lock also constrains the GR00T install.
COPY example_workflows/v2d_box_lift_posttraining/docker/requirements-learner.lock /opt/requirements-learner.lock

# Keep the simulator's Python, torch and numpy unchanged.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.11.13 && uv venv --python 3.11.13 /opt/gr00t-venv \
    && git init /opt/Isaac-GR00T \
    && git -C /opt/Isaac-GR00T remote add origin https://github.com/NVIDIA/Isaac-GR00T.git \
    && git -C /opt/Isaac-GR00T fetch --depth 1 origin ab88b50c718f6528e1df9dcbaf75865d1b604760 \
    && GIT_LFS_SKIP_SMUDGE=1 git -C /opt/Isaac-GR00T checkout --detach FETCH_HEAD \
    && uv pip install --python /opt/gr00t-venv/bin/python --no-deps \
       --only-binary :all: --no-binary gr00t -c /opt/requirements-learner.lock /opt/Isaac-GR00T
# PyPI is searched first, so only the +cu128 torch builds come from the PyTorch index;
# omegaconf's antlr4 runtime ships no wheel.
RUN uv pip install --python /opt/gr00t-venv/bin/python --require-hashes \
    --only-binary :all: --no-binary antlr4-python3-runtime \
    --index https://pypi.org/simple --default-index https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-first-match -r /opt/requirements-learner.lock

# Upstream copies source/ and scripts/; the joint-policy integration also needs these modules.
COPY --from=v2d_source .git /workspace/video_to_data/.git
COPY --from=v2d_source robotic_grounding/groot_finetune /workspace/video_to_data/robotic_grounding/groot_finetune
RUN git config --system --add safe.directory /workspace/video_to_data
ARG REPO=/workspace/posttrain
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ${REPO}/
COPY isaac_rl_posttraining ${REPO}/isaac_rl_posttraining
COPY simulators ${REPO}/simulators
COPY scripts ${REPO}/scripts
COPY example_workflows ${REPO}/example_workflows
WORKDIR /workspace/posttrain
# --no-config: the image pins its own learner stack; the repository's host-side uv settings do not apply.
RUN /usr/local/bin/python -m pip install \
       -r /workspace/posttrain/example_workflows/v2d_box_lift_posttraining/requirements-collision.txt \
    && uv pip install --python /opt/gr00t-venv/bin/python --no-config \
       --only-binary :all: --no-binary isaac-rl-posttraining \
       -c /opt/requirements-learner.lock /workspace/posttrain \
    && /opt/gr00t-venv/bin/python -c 'from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7; \
       import example_workflows.v2d_box_lift_posttraining.launch'

ARG POSTTRAIN_REVISION
RUN test -n "$POSTTRAIN_REVISION"
LABEL org.opencontainers.image.revision=$POSTTRAIN_REVISION
ENV POSTTRAIN_SOURCE_REVISION=$POSTTRAIN_REVISION \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface XDG_CACHE_HOME=/cache/xdg \
    WANDB_CACHE_DIR=/cache/wandb WANDB_DATA_DIR=/cache/wandb-data
ENTRYPOINT ["/opt/gr00t-venv/bin/python", "-m", "example_workflows.v2d_box_lift_posttraining.docker_entrypoint"]
