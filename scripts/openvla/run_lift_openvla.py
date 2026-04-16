#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the standard OpenVLA model on the Isaac-Lift-Cube-Franka task.

The model receives a 256×256 RGB image from the fixed table camera and the
natural-language instruction "lift the cube", then outputs a 7-DoF delta
end-effector action that is fed directly to the IK-relative action space.

Usage
-----
    isaaclab.bat -p scripts/openvla/run_lift_openvla.py --num_envs 1

Optional flags
--------------
    --num_steps INT     Total sim steps to run per launch (default: 500)
    --num_episodes INT  Stop after this many completed episodes (default: inf)
    --unnorm_key STR    OpenVLA dataset statistics key for action denormalization
                        (default: bridge_orig).  Other options: fractal20220817_data
    --headless          Run without GUI (passed through to AppLauncher)
"""

import argparse

# ── AppLauncher MUST be created before any Isaac Sim / IsaacLab imports ────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenVLA × Isaac-Lift-Cube-Franka runner")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_steps", type=int, default=500, help="Total simulation steps to run")
parser.add_argument("--num_episodes", type=int, default=0, help="Stop after N episodes (0 = unlimited)")
parser.add_argument(
    "--unnorm_key",
    type=str,
    default="bridge_orig",
    help="OpenVLA unnormalization key (dataset statistics). Default: bridge_orig",
)
parser.add_argument(
    "--action_repeat",
    type=int,
    default=5,
    help="Repeat each OpenVLA action for this many env steps before querying again (default: 5). "
         "Reduces inference load: at 50 Hz policy rate, repeat=5 → ~10 Hz effective VLA rate.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature for OpenVLA token generation. 0.0 = greedy (do_sample=False). "
         "Values > 0 enable stochastic sampling (do_sample=True), which produces varied actions "
         "when greedy decoding collapses to a constant output due to sim-to-real gap. "
         "Recommended starting value: 0.5 (default: 0.0).",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=1.0,
    help="Multiply the arm delta (6-DoF) by this factor after clamping. Used to diagnose whether "
         "the IK controller responds to the action at all — e.g. 100.0 amplifies sub-millimetre "
         "raw actions to centimetre-scale commands. Gripper is not scaled. (default: 1.0)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Safe to import Isaac Sim / IsaacLab now ────────────────────────────────────
import gc
import torch
from PIL import Image

import numpy as np
import gymnasium as gym  # noqa: F401 — registers tasks on import

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.lift  # noqa: F401 — registers lift envs

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.openvla_env_cfg import (
    FrankaCubeLiftOpenVLAEnvCfg,
)


# ── OpenVLA helpers ────────────────────────────────────────────────────────────

def load_openvla(device: str = "cuda:0"):
    """Load the OpenVLA-7B model and processor onto *device*."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("[OpenVLA] Loading processor from openvla/openvla-7b …")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    print("[OpenVLA] Loading model weights …")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    print("[OpenVLA] Model ready.")
    return model, processor


# Maximum delta translation (m) and rotation (rad) per step — clamps runaway actions.
_MAX_TRANS = 0.05  # 5 cm
_MAX_ROT = 0.2     # ~11 deg


def query_openvla(
    env: ManagerBasedRLEnv,
    model,
    processor,
    instruction: str,
    unnorm_key: str,
    camera_name: str = "camera",
    temperature: float = 0.0,
    action_scale: float = 1.0,
) -> torch.Tensor:
    """Query OpenVLA with the current camera frame and return an action tensor.

    Parameters
    ----------
    env:           The running IsaacLab environment.
    model:         OpenVLA AutoModelForVision2Seq instance.
    processor:     Matching AutoProcessor instance.
    instruction:   Natural-language command, e.g. "lift the cube".
    unnorm_key:    Dataset statistics key for action denormalization.
    camera_name:   Key used to look up the camera sensor in env.scene.
    temperature:   Sampling temperature. 0.0 = greedy decoding. > 0 enables
                   stochastic sampling, which breaks constant-output collapse
                   caused by the sim-to-real gap in OpenVLA's visual encoder.

    Returns
    -------
    action : torch.Tensor, shape (num_envs, 7)
        Columns 0-5 — delta end-effector pose (dx, dy, dz, droll, dpitch, dyaw).
        Column  6   — binary gripper (1.0 = open, 0.0 = close).
    """
    # 1. Grab RGB from env 0 only (OpenVLA is queried once; action is tiled).
    #    Camera output shape: (num_envs, H, W, 4) or (num_envs, H, W, 3).
    #    Dtype may be uint8 (0-255) or float32 (0-1) depending on Isaac Sim version.
    #    NOTE: camera read is NOT inside torch.inference_mode() — the decorator was
    #    removed to avoid any OmniGraph/CUDA tensor aliasing issues (see openvla#305).
    rgb_hwc = env.scene[camera_name].data.output["rgb"][0, ..., :3]  # (H, W, 3)
    rgb_np = rgb_hwc.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        # float32 / float16 — scale to [0, 255] uint8 for PIL
        rgb_np = (rgb_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    # Diagnostic: print image stats every query so we can verify the camera is live.
    if not hasattr(query_openvla, "_frame_count"):
        query_openvla._frame_count = 0
    query_openvla._frame_count += 1
    print(
        f"  [camera] frame={query_openvla._frame_count:4d} "
        f"mean={rgb_np.mean():.2f}  std={rgb_np.std():.2f}  "
        f"min={int(rgb_np.min())}  max={int(rgb_np.max())}"
    )
    # Overwrite openvla_frame_0.png on every query so the IDE shows a live view.
    Image.fromarray(rgb_np).save("openvla_frame_0.png")

    pil_img = Image.fromarray(rgb_np).resize((256, 256))

    # 2. Run OpenVLA inference.
    # Match the reference call pattern exactly (openvla/openvla#305):
    #   inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    # BatchFeature.to(dtype) only casts floating-point tensors (pixel_values →
    # bfloat16) and leaves integer tensors (input_ids, attention_mask) as int64.
    device = next(model.parameters()).device
    inputs = processor(instruction, pil_img).to(device, dtype=torch.bfloat16)
    # Greedy decoding (temperature=0) collapses to a constant action when the
    # model's visual encoder maps all sim images to the same feature vector
    # (sim-to-real gap).  Stochastic sampling (temperature > 0) breaks this.
    do_sample = temperature > 0.0
    sample_kwargs = {"temperature": temperature} if do_sample else {}
    with torch.no_grad():
        raw_action = model.predict_action(
            **inputs, unnorm_key=unnorm_key, do_sample=do_sample, **sample_kwargs
        )
    # raw_action: np.ndarray (7,) — [dx, dy, dz, droll, dpitch, dyaw, gripper]

    # Free temporary inference tensors before env.step() needs VRAM for RTX rendering.
    del inputs
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 3. Package into IsaacLab action tensor (num_envs, 7).
    #    Clamp translations and rotations, then optionally scale for diagnostics.
    #    Scale is applied AFTER the clamp so the clamp still prevents pre-scale
    #    runaway; at default action_scale=1.0 behaviour is unchanged.
    raw_action[:3] = np.clip(raw_action[:3], -_MAX_TRANS, _MAX_TRANS)
    raw_action[3:6] = np.clip(raw_action[3:6], -_MAX_ROT, _MAX_ROT)
    if action_scale != 1.0:
        raw_action[:6] = raw_action[:6] * action_scale

    #    arm_action (6,) — delta EE pose, tiled across all envs.
    arm = torch.tensor(raw_action[:6], dtype=torch.float32, device=env.device)
    arm = arm.unsqueeze(0).expand(env.num_envs, -1)  # (N, 6)

    #    gripper (1,) — threshold: positive = open (1.0), non-positive = close (0.0).
    gripper_val = 1.0 if raw_action[6] > 0.0 else 0.0
    gripper = torch.full((env.num_envs, 1), gripper_val, dtype=torch.float32, device=env.device)

    return torch.cat([arm, gripper], dim=-1), raw_action  # (N, 7), (7,)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Step 1: create and fully initialise the environment FIRST ─────────────
    # Isaac Sim needs to claim its GPU memory (physics, rendering) before the
    # large OpenVLA model is loaded.  Loading the model first leaves too little
    # VRAM for the PhysX forward pass in env.reset() and causes a native crash.
    cfg = FrankaCubeLiftOpenVLAEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)

    print("[Runner] Initialising environment …")
    obs, _ = env.reset()
    # num_rerenders_on_reset=1 (set in openvla_env_cfg) ensures the camera
    # buffer is populated during reset — no manual warmup steps needed.
    print("[Runner] Environment ready.")

    # ── Step 2: load OpenVLA into whatever VRAM remains ───────────────────────
    model, processor = load_openvla(device="cuda:0")

    instruction = "place the red cube on other side of table"
    print(f'\n[Runner] Instruction: "{instruction}"')
    sampling_mode = f"temperature={args_cli.temperature}" if args_cli.temperature > 0.0 else "greedy"
    print(f"[Runner] Running for {args_cli.num_steps} steps "
          f"(action_repeat={args_cli.action_repeat}, sampling={sampling_mode}) …\n")

    print(f"[Runner] Env action space: {env.action_space}")
    print(f"[Runner] Action manager total dim: {env.action_manager.total_action_dim}\n")

    episode_count = 0
    episode_reward = 0.0
    cached_action: torch.Tensor | None = None  # scaled delta, applied every step
    last_raw: np.ndarray | None = None

    for step in range(args_cli.num_steps):
        # Query OpenVLA every action_repeat steps.
        # The arm delta is divided by action_repeat so that applying the same action
        # for action_repeat consecutive steps totals exactly one full VLA delta —
        # no accumulation, no hold-then-cancel.
        if cached_action is None or step % args_cli.action_repeat == 0:
            delta_action, last_raw = query_openvla(
                env, model, processor,
                instruction=instruction,
                unnorm_key=args_cli.unnorm_key,
                temperature=args_cli.temperature,
                action_scale=args_cli.action_scale,
            )
            # Scale arm (6D) down; gripper (1D) is binary so leave it unscaled.
            delta_action = delta_action.clone()
            delta_action[:, :6] /= args_cli.action_repeat
            cached_action = delta_action

        obs, reward, terminated, truncated, info = env.step(cached_action)

        episode_reward += reward[0].item()

        if cached_action is None or step % args_cli.action_repeat == 0:
            # Print with full 6-decimal precision so sub-millimetre action changes
            # are visible.  If raw_action is EXACTLY identical across queries the
            # model is outputting the same token sequence for every image (sim-to-real
            # gap) rather than there being a code bug.
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
            cached_action = None   # force a fresh query at the start of the new episode
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
