# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

variable "V2D_SOURCE" { default = "" }
variable "POSTTRAIN_IMAGE" { default = "posttrain-v2d:local" }
variable "POSTTRAIN_REVISION" { default = "" }

target "v2d" {
  context = "${V2D_SOURCE}/robotic_grounding"
  dockerfile = "workflow/Dockerfile"
}

target "training" {
  context = "."
  dockerfile = "example_workflows/v2d_box_lift_posttraining/docker/Dockerfile"
  contexts = {
    "v2d_base:bake" = "target:v2d"
    v2d_source = V2D_SOURCE
  }
  args = { POSTTRAIN_REVISION = POSTTRAIN_REVISION }
  tags = [POSTTRAIN_IMAGE]
}
