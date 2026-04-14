# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA environment config for the Franka cube-lift task.

Extends the IK-relative config with a fixed table camera so the
runner script can pass 256×256 RGB frames to OpenVLA.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import ik_rel_env_cfg


@configclass
class FrankaCubeLiftOpenVLAEnvCfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
    """Franka cube-lift with a table camera added for OpenVLA visual input.

    Action space: IK-relative delta end-effector pose (6-DoF) + binary gripper (1)
    — matches OpenVLA's 7-DoF delta EE output exactly.
    """

    def __post_init__(self):
        super().__post_init__()

        # Silence observation noise during VLA inference.
        self.observations.policy.enable_corruption = False

        # Longer episodes to give the VLA model time to complete the task.
        self.episode_length_s = 15.0

        # Re-render the RTX camera once after each reset so its buffer is populated
        # before the runner calls query_openvla().  Default is 0 (no re-render).
        self.num_rerenders_on_reset = 1

        # Fixed overhead-angled camera looking at the workspace.
        # Position: 1 m in front of robot base, 0.4 m up.
        # Rotation (ROS convention): roughly 45-deg downward look toward the table.
        # These intrinsics (focal_length / horizontal_aperture) match the stack
        # visuomotor task camera and give a reasonable FoV over the workspace.
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )
