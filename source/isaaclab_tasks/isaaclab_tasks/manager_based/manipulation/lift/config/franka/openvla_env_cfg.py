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

        # Disable command-pose debug visualisation — its per-step marker updates
        # trigger repeated IMemoryBudgetManagerFactory acquisitions and spam the log.
        self.commands.object_pose.debug_vis = False

        # Longer episodes to give the VLA model time to complete the task.
        self.episode_length_s = 15.0

        # Re-render the RTX camera once after each reset so its buffer is populated
        # before the runner calls query_openvla().  Default is 0 (no re-render).
        self.num_rerenders_on_reset = 1

        # Replace the default bright grey dome light with a dark-background variant.
        # visible_in_primary_ray=False renders the sky as pure black, giving strong
        # contrast against the white robot arm.  A slightly cooler colour (0.6, 0.7, 1.0)
        # and reduced intensity keep the scene adequately lit without a bright sky wash.
        from isaaclab.assets import AssetBaseCfg
        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(
                color=(0.6, 0.7, 1.0),
                intensity=1500.0,
                visible_in_primary_ray=False,
            ),
        )

        # Fixed camera looking from in front of the robot back toward the workspace.
        # The forward direction of this quaternion is (-0.866, 0, -0.5) — i.e. looking
        # in the -x/world direction at ~30° below horizontal.
        #
        # Position: 1.5 m in front of robot base, 0.7 m up.
        #   - Pulled back from 1.0 m so the full arm (EE at z≈0.3-0.5) is visible.
        #   - Raised from 0.4 m so the arm appears near the centre of the frame.
        # FOV: horizontal_aperture 27.7 → ~60° (was 20.955 → ~47°), wider to capture
        #   the whole workspace from cube (z≈0.05) to gripper (z≈0.5).
        # Clipping: far plane extended to 3.0 m so nothing is culled.
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,  # update every physics step — avoids timer-drift frame freezes
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=27.7,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.5, 0.0, 0.7),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )
