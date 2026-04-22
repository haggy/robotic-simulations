#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the OpenVLA-OFT model on the Isaac-Lift-Cube-Franka task.

OpenVLA-OFT replaces the base model's discrete token decoder with a continuous
L1-regression action head and adds action chunking: the policy produces a
sequence of N future actions per forward pass, which are executed open-loop
before the next query.  This yields ~26× faster action generation and ~3×
lower latency compared to the base tokenized OpenVLA.

Differences from run_lift_openvla.py
--------------------------------------
* Model loading uses openvla-oft utility functions (get_vla / get_action_head /
  get_processor) instead of AutoModelForVision2Seq.from_pretrained.
* action_repeat is replaced by action chunking via a deque: the model is
  queried once per chunk and individual actions are executed one step at a time.
* Gripper post-processing: normalize_gripper_action maps OFT's [0,1] output to
  [-1,+1]; no invert_gripper_action (IsaacLab uses positive = open, unlike LIBERO).
* Image resize to 224×224 is handled inside get_vla_action via
  prepare_images_for_vla (no manual PIL.resize needed).

Usage
-----
    isaaclab.bat -p scripts/openvla/run_lift_openvla_oft.py --num_envs 1

Optional flags
--------------
    --checkpoint STR           HuggingFace model ID or local directory path
                               (default: moojink/openvla-7b-oft-finetuned-libero-spatial)
    --unnorm_key STR           Action un-normalization key from dataset_statistics.json.
                               Leave empty to auto-detect (tries "_no_noops" variant first).
                               Example: libero_spatial_no_noops
    --num_steps INT            Total sim steps (default: 500)
    --num_episodes INT         Stop after N episodes; 0 = unlimited (default: 0)
    --num_open_loop_steps INT  Actions per chunk; 0 = use NUM_ACTIONS_CHUNK from
                               openvla-oft/prismatic/vla/constants.py (default: 0)
    --center_crop              Center-crop images before model input.  Enable if the
                               checkpoint was trained with random crop augmentation.
    --use_proprio              Include EEF pose + gripper qpos as 8-D proprioception.
                               Must match the checkpoint training config.
    --action_scale FLOAT       Multiply arm delta after clamping (default: 1.0)
    --headless                 Run without GUI
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# ── AppLauncher MUST be created before any Isaac Sim / IsaacLab imports ────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenVLA-OFT × Isaac-Lift-Cube-Franka runner")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_steps", type=int, default=500, help="Total simulation steps to run")
parser.add_argument("--num_episodes", type=int, default=0, help="Stop after N episodes (0 = unlimited)")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    help="HuggingFace model ID or local directory containing OFT checkpoint files.",
)
parser.add_argument(
    "--unnorm_key",
    type=str,
    default="",
    help="Action un-normalization key from the checkpoint's dataset_statistics.json. "
         "Leave empty to auto-detect (prefers the '_no_noops' variant when available). "
         "Example: libero_spatial_no_noops",
)
parser.add_argument(
    "--num_open_loop_steps",
    type=int,
    default=0,
    help="Number of actions to execute open-loop per model query (the chunk size). "
         "0 = use NUM_ACTIONS_CHUNK from prismatic/vla/constants.py (8 for LIBERO checkpoints).",
)
parser.add_argument(
    "--center_crop",
    action="store_true",
    help="Apply a center crop to images before feeding to the model. "
         "Use this if the checkpoint was trained with random crop image augmentation.",
)
parser.add_argument(
    "--use_proprio",
    action="store_true",
    help="Include end-effector position, axis-angle orientation, and gripper qpos "
         "as an 8-D proprioception vector. Must match the checkpoint's training config.",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=1.0,
    help="Multiply the arm delta (6-DoF) by this factor after clamping. "
         "Useful for diagnosing IK controller responsiveness. Gripper is not scaled. (default: 1.0)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Safe to import Isaac Sim / IsaacLab now ────────────────────────────────────
import gc
from collections import deque

import numpy as np
import torch
from PIL import Image

import gymnasium as gym  # noqa: F401 — registers tasks on import
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.lift  # noqa: F401 — registers lift envs

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.openvla_oft_env_cfg import (
    FrankaCubeLiftOpenVLAOFTEnvCfg,
)

# ── Add openvla-oft repo to Python path ───────────────────────────────────────
# openvla-oft lives at <IsaacLab root>/openvla-oft.  All OFT helper modules
# (experiments.robot.*, prismatic.*) are imported relative to this root.
_OFT_ROOT = Path(__file__).resolve().parents[2] / "openvla-oft"
if str(_OFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFT_ROOT))


# ── OFT model config ───────────────────────────────────────────────────────────

@dataclass
class OFTRunCfg:
    """Minimal config accepted by openvla-oft utility functions.

    Fields mirror the relevant subset of GenerateConfig from run_libero_eval.py.
    Only the fields actually read by get_vla, get_action_head, get_proprio_projector,
    get_processor, and get_vla_action are included here.
    """
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    # Vision backbone
    use_film: bool = False
    num_images_in_input: int = 1       # single third-person camera; 2 = add wrist cam
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # Action head
    use_l1_regression: bool = True     # OFT uses L1 regression by default
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    # Chunking
    num_open_loop_steps: int = 8
    # Image preprocessing
    center_crop: bool = False
    # Action un-normalization
    unnorm_key: str = ""
    # Proprioception
    use_proprio: bool = False
    # LoRA (only needed when use_film=True to load the PeftModel adapter)
    lora_rank: int = 32


# ── Loading helpers ────────────────────────────────────────────────────────────

def load_oft_components(cfg: OFTRunCfg):
    """Load VLA backbone, action head, (optional) proprio projector, and processor.

    The VLA is moved to DEVICE (cuda:0) inside get_vla.
    Action head and proprio projector are also placed on DEVICE inside their
    respective loaders.
    """
    from experiments.robot.openvla_utils import (
        get_vla,
        get_action_head,
        get_proprio_projector,
        get_processor,
    )

    print("[OFT] Loading VLA backbone …")
    vla = get_vla(cfg)

    print("[OFT] Loading action head (L1 regression) …")
    action_head = get_action_head(cfg, vla.llm_dim)

    proprio_projector = None
    if cfg.use_proprio:
        print("[OFT] Loading proprioception projector …")
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=8)

    print("[OFT] Loading processor …")
    processor = get_processor(cfg)

    # Resolve unnorm_key: auto-detect if not provided.
    if not cfg.unnorm_key:
        available = list(vla.norm_stats.keys())
        cfg.unnorm_key = available[0] if available else ""
        for key in available:
            if "no_noops" in key:
                cfg.unnorm_key = key
                break
    assert cfg.unnorm_key in vla.norm_stats, (
        f"unnorm_key '{cfg.unnorm_key}' not found in model norm_stats. "
        f"Available keys: {list(vla.norm_stats.keys())}"
    )
    print(f"[OFT] Resolved unnorm_key: {cfg.unnorm_key}")
    print("[OFT] Model components ready.")

    return vla, action_head, proprio_projector, processor


# ── Inference helpers ──────────────────────────────────────────────────────────

_MAX_TRANS = 0.05   # 5 cm — clamps runaway translation per step
_MAX_ROT   = 0.2    # ~11 deg — clamps runaway rotation per step


def query_oft(
    env: ManagerBasedRLEnv,
    cfg: OFTRunCfg,
    vla,
    processor,
    action_head,
    proprio_projector,
    instruction: str,
    camera_name: str = "camera",
) -> list:
    """Query OpenVLA-OFT with the current camera frame and return an action chunk.

    The function builds an observation dict, calls get_vla_action (which handles
    image resize to 224×224, optional center crop, and the OFT forward pass), then
    frees temporary VRAM before the next env.step() RTX rendering call.

    Parameters
    ----------
    env:               Running IsaacLab environment.
    cfg:               OFTRunCfg with unnorm_key, center_crop, use_proprio etc.
    vla:               Loaded OFT VLA backbone.
    processor:         Matching HuggingFace processor.
    action_head:       L1-regression action head.
    proprio_projector: Proprioception projector or None.
    instruction:       Natural-language task instruction.
    camera_name:       Key used to look up the camera sensor in env.scene.

    Returns
    -------
    List of np.ndarray (7,) — one per open-loop step in the chunk.
    Each element: [dx, dy, dz, droll, dpitch, dyaw, gripper_raw].
    gripper_raw is in [0,1] range (OFT dataset encoding: 0=close, 1=open).
    """
    from experiments.robot.openvla_utils import get_vla_action

    # 1. Get RGB from env 0 only (OFT is queried once; action chunk is tiled).
    #    Camera output shape: (num_envs, H, W, 4) or (num_envs, H, W, 3).
    #    Dtype may be uint8 (0-255) or float32 (0-1) depending on Isaac Sim version.
    rgb_hwc = env.scene[camera_name].data.output["rgb"][0, ..., :3]  # (H, W, 3)
    rgb_np = rgb_hwc.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    # Diagnostic: print frame statistics so we can verify the camera is live.
    if not hasattr(query_oft, "_frame_count"):
        query_oft._frame_count = 0
    query_oft._frame_count += 1
    print(
        f"  [camera] frame={query_oft._frame_count:4d}  "
        f"mean={rgb_np.mean():.2f}  std={rgb_np.std():.2f}  "
        f"min={int(rgb_np.min())}  max={int(rgb_np.max())}"
    )
    # Overwrite openvla_frame_0.png on every query for a live IDE preview.
    Image.fromarray(rgb_np).save("openvla_frame_0.png")

    # 2. Build observation dict.
    #    get_vla_action expects "full_image" as a (H, W, 3) uint8 numpy array.
    #    It internally resizes to 224×224 and applies center_crop if cfg.center_crop.
    obs: dict = {"full_image": rgb_np}

    if cfg.use_proprio:
        # 8-D proprioception: eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2)
        # Isaac Sim quaternion convention: (w, x, y, z); scipy uses (x, y, z, w).
        from scipy.spatial.transform import Rotation

        ee_pos = env.scene["ee_frame"].data.target_pos_w[0, 0, :3].cpu().numpy()     # (3,)
        ee_quat_wxyz = env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()  # (4,) w,x,y,z
        ee_axis_angle = Rotation.from_quat(
            [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]]
        ).as_rotvec()  # (3,)
        finger_idx = env.scene["robot"].find_joints("panda_finger.*")[0]
        gripper_qpos = env.scene["robot"].data.joint_pos[0, finger_idx].cpu().numpy()  # (2,)
        obs["state"] = np.concatenate([ee_pos, ee_axis_angle, gripper_qpos])  # (8,)

    # 3. Run OFT inference.
    #    get_vla_action wraps the forward pass in torch.inference_mode() internally.
    #    Returns a list of num_open_loop_steps numpy arrays, each shape (7,).
    actions = get_vla_action(
        cfg=cfg,
        vla=vla,
        processor=processor,
        obs=obs,
        task_label=instruction,
        action_head=action_head,
        proprio_projector=proprio_projector,
    )

    # 4. Free temporary inference tensors before env.step() needs VRAM for RTX rendering.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return actions  # List[np.ndarray (7,)]


def process_action_for_isaaclab(
    raw_action: np.ndarray,
    env: ManagerBasedRLEnv,
    action_scale: float = 1.0,
) -> torch.Tensor:
    """Convert one OFT action (7,) into an IsaacLab action tensor (num_envs, 7).

    Gripper convention
    ------------------
    OFT dataset encoding:           0 = close,  1 = open
    normalize_gripper_action maps:  0 → -1,     1 → +1  (binarized)
    IsaacLab BinaryJointPositionActionCfg: positive → open

    Therefore we apply normalize_gripper_action but NOT invert_gripper_action.
    invert_gripper_action is for the LIBERO simulator which uses -1=open, +1=close —
    the opposite of IsaacLab's convention.

    Parameters
    ----------
    raw_action:   (7,) numpy array from get_vla_action.
    env:          Running IsaacLab environment (provides device + num_envs).
    action_scale: Multiplier applied to arm delta after clamping.

    Returns
    -------
    torch.Tensor, shape (num_envs, 7).
    Columns 0-5: clamped + scaled delta EE pose (dx, dy, dz, droll, dpitch, dyaw).
    Column  6:   binary gripper (1.0 = open, 0.0 = close).
    """
    from experiments.robot.robot_utils import normalize_gripper_action

    # Normalize gripper [0,1] → [-1,+1] binarized; leave arm dims unchanged.
    action = normalize_gripper_action(raw_action, binarize=True)

    # Clamp arm deltas, then optionally scale for diagnostics.
    action[:3] = np.clip(action[:3], -_MAX_TRANS, _MAX_TRANS)
    action[3:6] = np.clip(action[3:6], -_MAX_ROT, _MAX_ROT)
    if action_scale != 1.0:
        action[:6] = action[:6] * action_scale

    # Arm delta (6,) tiled across all parallel environments.
    arm = torch.tensor(action[:6], dtype=torch.float32, device=env.device)
    arm = arm.unsqueeze(0).expand(env.num_envs, -1)  # (N, 6)

    # Gripper: +1 (open after normalize) → 1.0, -1 (close) → 0.0.
    gripper_val = 1.0 if action[6] > 0.0 else 0.0
    gripper = torch.full((env.num_envs, 1), gripper_val, dtype=torch.float32, device=env.device)

    return torch.cat([arm, gripper], dim=-1)  # (N, 7)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Step 1: create and fully initialise the environment FIRST ─────────────
    # Isaac Sim must claim its GPU memory (physics + rendering) before the
    # large VLA model is loaded.  Loading the model first can leave too little
    # VRAM for the PhysX forward pass in env.reset() and cause a native crash.
    cfg = FrankaCubeLiftOpenVLAOFTEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)

    print("[Runner] Initialising environment …")
    obs, _ = env.reset()
    # num_rerenders_on_reset=1 (inherited from FrankaCubeLiftOpenVLAEnvCfg) ensures
    # the camera buffer is populated after reset — no manual warmup steps needed.
    print("[Runner] Environment ready.")

    # ── Step 2: load OFT model components ─────────────────────────────────────
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    num_open_loop = (
        args_cli.num_open_loop_steps if args_cli.num_open_loop_steps > 0 else NUM_ACTIONS_CHUNK
    )

    oft_cfg = OFTRunCfg(
        pretrained_checkpoint=args_cli.checkpoint,
        unnorm_key=args_cli.unnorm_key,
        center_crop=args_cli.center_crop,
        use_proprio=args_cli.use_proprio,
        num_open_loop_steps=num_open_loop,
    )

    vla, action_head, proprio_projector, processor = load_oft_components(oft_cfg)

    instruction = "place the red cube on other side of table"
    print(f'\n[Runner] Instruction: "{instruction}"')
    print(f"[Runner] Checkpoint:            {args_cli.checkpoint}")
    print(f"[Runner] unnorm_key:            {oft_cfg.unnorm_key}")
    print(f"[Runner] Chunk size (open-loop):{oft_cfg.num_open_loop_steps}")
    print(f"[Runner] center_crop:           {args_cli.center_crop}")
    print(f"[Runner] use_proprio:           {args_cli.use_proprio}")
    print(f"[Runner] Running for {args_cli.num_steps} steps …\n")

    print(f"[Runner] Env action space: {env.action_space}")
    print(f"[Runner] Action manager total dim: {env.action_manager.total_action_dim}\n")

    episode_count = 0
    episode_reward = 0.0
    action_queue: deque = deque()
    last_raw: np.ndarray | None = None

    for step in range(args_cli.num_steps):
        # Query OFT when the action deque runs dry.
        queried = False
        if len(action_queue) == 0:
            chunk = query_oft(
                env, oft_cfg, vla, processor, action_head, proprio_projector, instruction
            )
            action_queue.extend(chunk)
            queried = True

        raw_action = action_queue.popleft()  # (7,) — gripper in [0,1]
        last_raw = raw_action
        action_tensor = process_action_for_isaaclab(raw_action, env, args_cli.action_scale)

        obs, reward, terminated, truncated, info = env.step(action_tensor)
        episode_reward += reward[0].item()

        if queried:
            np.set_printoptions(precision=6, suppress=True)
            ee_pos = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()  # (3,)
            print(
                f"  step {step:5d} | reward {reward[0].item():+.4f} | "
                f"ee_pos={ee_pos} | raw_action={last_raw}"
            )

        done = terminated | truncated  # (N,)
        if done.any():
            episode_count += 1
            print(f"  [Episode {episode_count} done] return = {episode_reward:.3f}")
            episode_reward = 0.0
            action_queue.clear()  # discard stale chunk — next step re-queries
            last_raw = None
            obs, _ = env.reset()

            if args_cli.num_episodes > 0 and episode_count >= args_cli.num_episodes:
                print(f"[Runner] Reached {args_cli.num_episodes} episodes. Stopping.")
                break

    env.close()
    print("[Runner] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
