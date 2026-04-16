# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA environment config for the Franka cube-lift task.

Extends the IK-relative config with a fixed table camera so the
runner script can pass 256×256 RGB frames to OpenVLA.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs.common import ViewerCfg

from . import ik_rel_env_cfg


@configclass
class FrankaCubeLiftOpenVLAEnvCfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
    """Franka cube-lift with a table camera added for OpenVLA visual input.

    Action space: IK-relative delta end-effector pose (6-DoF) + binary gripper (1)
    — matches OpenVLA's 7-DoF delta EE output exactly.
    """

    def __post_init__(self):
        super().__post_init__()

        # Position the UI viewport camera close to the robot so the workspace is
        # immediately visible on launch instead of the default (7.5, 7.5, 7.5).
        self.viewer = ViewerCfg(
            eye=(1.5, -1.5, 1.5),    # front-right-above the table
            lookat=(0.5, 0.0, 0.3),  # centre of the robot workspace
        )

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

        # Override cube with a solid red visual material so it stands out clearly
        # against the dark background.  All physics properties (scale, rigid body
        # solver settings, init state) are copied from joint_pos_env_cfg unchanged.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # pure red
                    roughness=0.5,
                    metallic=0.0,
                ),
            ),
        )

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

        # Camera rotated 45° CCW around world Z from the previous front-left position,
        # then pulled 20% further back along the same sight line.
        #
        # Look-at: (0.45, 0.05, 0.15) — centre of the cube workspace area
        # Position: (1.085, -0.130, 0.841) — 1.2× the (0.979, -0.100, 0.726)
        #   cam→look-at vector; same direction, 20% further back
        #
        # Quaternion unchanged (same viewing direction):
        # (w,x,y,z) = (0.29641, -0.74018, -0.56046, 0.22439)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,  # update every physics step — avoids timer-drift frame freezes
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=27.7,  # ~60° FOV — wide enough to capture arm + workspace
                clipping_range=(0.05, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.085, -0.130, 0.841),
                rot=(0.29641, -0.74018, -0.56046, 0.22439),
                convention="ros",
            ),
        )
