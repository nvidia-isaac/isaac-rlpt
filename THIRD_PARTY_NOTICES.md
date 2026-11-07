# Third-Party Notices

Isaac RL Post-Training is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
It contains, or downloads during setup, portions of the third-party software listed below.
Each entry gives the upstream project, the revision used, the files that contain it, what was changed, and the upstream copyright and license.

## Stable-Baselines3

- Upstream: https://github.com/DLR-RM/stable-baselines3 (`stable_baselines3/common/distributions.py`)
- Revision: release `v2.7.1` (commit `8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458`).
  The adapted code matches the upstream file as released in v2.5.0 through v2.7.1, which is identical across those four releases; v2.7.1 is the last of them.
- Files: `isaac_rl_posttraining/modules/base/distributions.py`
- Changes: trimmed to the diagonal Gaussian, tanh-squashed diagonal Gaussian and categorical distributions, the tanh bijector, `sum_independent_dims` and `kl_divergence`; the other distributions and the distribution factory were removed.
  The tanh change-of-variables term in the squashed Gaussian's log-probability is computed in a softplus form.
  Docstrings and type annotations were rewritten.
- Copyright: Copyright (c) 2019 Antonin Raffin
- License: MIT

```
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## TorchRL

- Upstream: https://github.com/pytorch/rl (`torchrl/record/loggers/wandb.py`)
- Revision: release `v0.6.0` (commit `7bf320c37b67328d9da17bc38a3fd723bd45cd6e`).
  The adapted code matches the upstream file as released in v0.3.0 through v0.6.0, which is identical across those releases; v0.6.0 is the last of them.
- Files: `isaac_rl_posttraining/loggers/wandb_logger.py`
- Changes: the logger was rebuilt on this package's own `Logger` base class; the video-tensor shape check, the scalar-plus-step logging and the default video frame rate are kept from upstream.
  Added: the run-directory rule (`wandb_run_dir`), resume-by-run-id handling, run id and URL accessors, dictionary, hyperparameter and artifact logging.
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates.
- License: MIT

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
```

## RLinf

- Upstream: https://github.com/RLinf/RLinf (the DreamZero integration under `rlinf/models/embodiment/dreamzero/` and `rlinf/data/datasets/dreamzero/`)
- Revision: branch `main` at commit `888c76fa1ead1d7cf28cd035ec4b64910a6b0cd3` (2026-06-18).
  This is the last upstream change to the DreamZero code before the port on 2026-07-28; that code did not change upstream between the two dates.
  The model loading (`get_model`), the batch collation (`DreamZeroCollator.collate_batch`), the inference transform (`DreamTransform` subclass) and the per-embodiment transform builders come from this revision.
- Files: `isaac_rl_posttraining/policies/dreamzero/layouts.py`, `model.py`, `transforms.py`
- Changes: the DreamZero model loading and input preprocessing were rewritten for in-process inference in this package.
  Model loading was trimmed of its distributed and supervised-training branches, and the embodiment layouts were collected into one specification per embodiment.
- Copyright: Copyright 2026 The RLinf Authors.
- License: Apache License 2.0, the same license as this repository; the full text is in [LICENSE](LICENSE).
  RLinf has no NOTICE file at this revision.

## robosuite

- Upstream: https://github.com/ARISE-Initiative/robosuite (`robosuite/utils/transform_utils.py`, function `quat2axisangle`)
- Revision: release `1.4.1` (the version this repository pins for LIBERO).
  The routine is identical in tag `v1.4.0` (commit `fbee5844ff5632f5b5698e204ec5357ca50be0df`), in the published 1.4.1 package, and in every later release through v1.5.2.
- Files: `simulators/libero/env.py` (`quat_to_axis_angle`), `simulators/libero/policies/pi0.py` (`_quat_to_axis_angle`)
- Changes: the input is copied rather than clipped in place, and the docstrings were rewritten; `env.py` also returns `float32`.
- Copyright: Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
- License: MIT

```
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This software includes the partial implementation of Deepmind Mujoco https://github.com/deepmind/mujoco. 
Deepmind Mujoco is licensed under the Apache License, Version 2.0 (the "License");
you may not use the files except in compliance with the License. 

You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
```

## PyTorch3D (downloaded during setup, not stored in this repository)

- Upstream: https://github.com/facebookresearch/pytorch3d (`pytorch3d/transforms/rotation_conversions.py`)
- Revision: tag `v0.7.9` (commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`)
- Where it lands: `example_workflows/isaaclab_gr00t_posttraining/scripts/fetch_externals.sh` writes it to `example_workflows/isaaclab_gr00t_posttraining/.externals/pytorch3d/` in the user's checkout, and `scripts/setup_env.sh` of the same workflow writes it to `$DEPS_DIR/pytorch3d/` inside the user's Isaac Lab container.
  Both scripts download PyTorch3D's `LICENSE` file at the same tag and save it as `pytorch3d/LICENSE` next to the code.
- Changes: the single file is installed as the `pytorch3d.transforms` package; its one import of `Device` from elsewhere in PyTorch3D is replaced by an equivalent local type alias.
  The scripts also generate a one-line package `__init__.py` and a minimal `pyproject.toml`.
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
- License: BSD 3-Clause

```
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
