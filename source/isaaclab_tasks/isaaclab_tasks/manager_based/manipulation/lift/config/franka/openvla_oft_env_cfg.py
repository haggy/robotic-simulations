# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA-OFT environment config for the Franka cube-lift task.

Inherits the table camera, red cube, dark dome light, and IK-relative action
space from FrankaCubeLiftOpenVLAEnvCfg.  The only override is a longer episode
budget to give the chunked policy time to complete the task.
"""

from isaaclab.utils import configclass

from . import openvla_env_cfg


@configclass
class FrankaCubeLiftOpenVLAOFTEnvCfg(openvla_env_cfg.FrankaCubeLiftOpenVLAEnvCfg):
    """Franka cube-lift with table camera, tuned for OpenVLA-OFT inference.

    Action space: IK-relative delta end-effector pose (6-DoF) + binary gripper (1)
    — identical to the base OpenVLA config; OFT outputs the same 7-DoF format.

    Camera, lighting, and cube visual material are inherited unchanged.
    Episode length is extended to 20 s to give the chunked policy more time.
    """

    def __post_init__(self):
        super().__post_init__()

        # OFT is faster per query (no token generation) and uses action chunking,
        # so a 20 s window lets a full task sequence complete without early timeout.
        self.episode_length_s = 20.0
