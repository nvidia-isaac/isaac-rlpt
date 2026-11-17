# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``isaac_rlpt`` library module Isaac Lab's ``train.py`` / ``play.py`` dispatch to (``--rl_library isaac_rlpt``).

Isaac Lab loads ``train_isaac_rlpt.py`` and ``play_isaac_rlpt.py`` by file path from the
linked module directory (the workflow's ``install.py``); the launch recipes in ``cli_args.py`` and the
launch logic in ``_launch.py`` import without Isaac Lab and are what the tests cover.
"""
