"""
consistency_fm_action_decoder.py
=================================
A design sketch for integrating Consistency-FM into OpenVLA-OFT as a
conditioned continuous action decoder.

Architecture overview
---------------------
OpenVLA-OFT's 4-layer MLP action head is replaced by a Consistency-FM
velocity network. The LLM (Llama 2 7B) produces a hidden state tensor of
shape [B, seq_len, 4096]. We mean-pool it along the sequence dimension to
get a single context vector per sample, then inject it into the velocity
network at every layer via FiLM (Feature-wise Linear Modulation).

FiLM conditioning works as follows: given a conditioning vector c, two
small linear projections produce a per-channel scale γ and shift β. These
are applied element-wise to each hidden layer's pre-activation:

    h_conditioned = γ(c) * h + β(c)

This is the same mechanism OFT already uses to modulate SigLIP/DINOv2
vision features with language embeddings (their FiLM+ variant), so the
pattern is proven in this exact codebase.

Training objective
------------------
The velocity consistency loss from Consistency-FM (Eq. 6 in the paper):

    L = ||f_θ(t, x_t) - f_θ-(t+Δt, x_{t+Δt})||² + α||v_θ(t, x_t) - v_θ-(t+Δt, x_{t+Δt})||²

where f_θ(t, x_t) = x_t + (1-t) * v_θ(t, x_t).

The EMA target network (θ-) stabilises training, analogous to how
consistency models use a stop-gradient target.

Sampling
--------
At inference, we start from Gaussian noise x_0 ~ N(0, I) and run 1-2
Euler steps through the learned straight flow to produce an action chunk.
With K=2 segments this looks like:

    x_0.5 = x_0   + 0.5 * v_θ¹(0,   x_0)
    x_1   = x_0.5 + 0.5 * v_θ²(0.5, x_0.5)

The result x_1 is the predicted action chunk.

Dependencies: torch, einops (optional, for clarity)
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  FiLM conditioning block
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation to a hidden state h, conditioned
    on context vector c.

        h_out = gamma(c) * h + beta(c)

    Args:
        cond_dim:   Dimensionality of the conditioning vector c (e.g. 4096
                    for Llama 2 7B hidden states, or a projected version).
        hidden_dim: Dimensionality of h (the velocity network hidden state).
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        # Two linear projections: one for scale, one for shift.
        # Initialise so that gamma starts near 1 and beta near 0,
        # meaning the network starts close to the unconditioned baseline.
        self.to_gamma = nn.Linear(cond_dim, hidden_dim)
        self.to_beta  = nn.Linear(cond_dim, hidden_dim)

        # Initialise weights near zero so FiLM starts as identity.
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)   # gamma ≈ 1 at init
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)   # beta  ≈ 0 at init

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, hidden_dim]   – velocity network hidden state
            c: [B, cond_dim]     – LLM conditioning vector
        Returns:
            h_out: [B, hidden_dim]
        """
        gamma = self.to_gamma(c)   # [B, hidden_dim]
        beta  = self.to_beta(c)    # [B, hidden_dim]
        return gamma * h + beta


# ---------------------------------------------------------------------------
# 2.  Timestep embedding  (sinusoidal, as in DDPM / flow matching literature)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """
    Encodes scalar flow timestep t ∈ [0, 1] into a learnable embedding.
    Uses sinusoidal features followed by a 2-layer MLP, matching the
    convention used in Consistency Models and many flow matching papers.

    Args:
        embed_dim: Output embedding dimensionality.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)   # [embed_dim/2]

        # Small MLP to project sinusoidal features → embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] scalar timesteps in [0, 1]
        Returns:
            emb: [B, embed_dim]
        """
        # Scale t to match the frequency convention (multiply by 1000 for
        # numerical range similar to diffusion model timestep embeddings).
        t_scaled = t * 1000.0                              # [B]
        args = t_scaled[:, None] * self.freqs[None, :]    # [B, embed_dim/2]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        return self.proj(emb)                              # [B, embed_dim]


# ---------------------------------------------------------------------------
# 3.  Velocity network  (the Consistency-FM core)
# ---------------------------------------------------------------------------

class ConsistencyFMVelocityNet(nn.Module):
    """
    Predicts the velocity v_θ(t, x_t | c) for a single flow segment.

    Architecture:
        input  : [x_t; t_emb]  concatenated → linear → N × (Linear + FiLM + ReLU) → output
        where t_emb is the sinusoidal timestep embedding and the FiLM layer at
        each residual block injects the LLM conditioning signal c.

    The output is v_θ ∈ R^action_dim, the velocity pointing from x_t toward x_1.

    Args:
        action_dim:  Dimensionality of the robot action vector (e.g. 7 for
                     a standard 6-DoF arm + gripper, or 7*chunk_size for
                     action chunking).
        hidden_dim:  Width of internal MLP layers.
        cond_dim:    Dimensionality of the LLM conditioning vector fed into
                     FiLM. Typically 4096 (Llama 2 7B) or a projected size.
        n_layers:    Number of residual blocks (default 4 matches OFT's MLP).
        t_embed_dim: Dimensionality of the timestep embedding.
    """

    def __init__(
        self,
        action_dim:  int,
        hidden_dim:  int  = 512,
        cond_dim:    int  = 4096,
        n_layers:    int  = 4,
        t_embed_dim: int  = 128,
    ):
        super().__init__()
        self.action_dim  = action_dim
        self.hidden_dim  = hidden_dim
        self.t_embed_dim = t_embed_dim

        # Timestep encoder
        self.t_embed = TimestepEmbedding(t_embed_dim)

        # Input projection: concatenate x_t and t embedding
        self.input_proj = nn.Linear(action_dim + t_embed_dim, hidden_dim)

        # Residual blocks: each is Linear → FiLM → ReLU
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(cond_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Output projection back to action space
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Initialise output near zero so early training doesn't produce
        # large random velocities that destabilise the flow.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,   # [B, action_dim]   noisy action at time t
        t:   torch.Tensor,   # [B]               flow timestep in [0, 1]
        c:   torch.Tensor,   # [B, cond_dim]     LLM conditioning vector
    ) -> torch.Tensor:
        """
        Returns velocity v_θ(t, x_t | c) with shape [B, action_dim].
        """
        # Encode timestep and concatenate with noisy action
        t_emb = self.t_embed(t)                            # [B, t_embed_dim]
        h = torch.cat([x_t, t_emb], dim=-1)               # [B, action_dim + t_embed_dim]
        h = self.input_proj(h)                             # [B, hidden_dim]

        # Residual blocks with FiLM conditioning at each layer
        for linear, film in zip(self.layers, self.film_layers):
            residual = h
            h = linear(h)          # linear transform
            h = film(h, c)         # FiLM: gamma(c)*h + beta(c)
            h = F.relu(h)
            h = h + residual       # residual connection for training stability

        return self.output_proj(h)                         # [B, action_dim]


# ---------------------------------------------------------------------------
# 4.  Multi-segment Consistency-FM action decoder
# ---------------------------------------------------------------------------

class ConsistencyFMActionDecoder(nn.Module):
    """
    Full Consistency-FM action decoder that replaces OpenVLA-OFT's 4-layer
    MLP output head.

    Supports:
        - Multi-segment training (K segments over [0,1], each with its own
          velocity network, as described in Section 3.3 of the paper).
        - EMA target network (θ-) for stable consistency loss computation.
        - 1-step or K-step inference.

    The LLM conditioning signal is extracted by mean-pooling the Llama 2
    hidden states across the sequence dimension, then optionally projecting
    to a smaller cond_dim for efficiency.

    Args:
        action_dim:   Dimensionality of the action vector (or chunk).
        llm_dim:      Hidden state dim of the LLM (4096 for Llama 2 7B).
        hidden_dim:   Width of velocity network hidden layers.
        cond_dim:     Projected conditioning dim fed into FiLM. Set equal
                      to llm_dim to skip projection.
        n_segments:   Number of flow segments K (1 = single global flow,
                      2 = piece-wise linear with 2 segments, etc.).
        n_layers:     Depth of each velocity network.
        ema_decay:    EMA decay for the target network θ-.
        alpha:        Weight on the velocity consistency term in the loss
                      (α in Eq. 6 of the paper).
    """

    def __init__(
        self,
        action_dim:  int,
        llm_dim:     int   = 4096,
        hidden_dim:  int   = 512,
        cond_dim:    int   = 256,
        n_segments:  int   = 2,
        n_layers:    int   = 4,
        ema_decay:   float = 0.999,
        alpha:       float = 1.0,
    ):
        super().__init__()
        self.action_dim  = action_dim
        self.n_segments  = n_segments
        self.ema_decay   = ema_decay
        self.alpha       = alpha

        # Optional projection: compress 4096-dim LLM hidden state to cond_dim.
        # This reduces FiLM parameter count and prevents the small velocity net
        # from being overwhelmed by the large LLM conditioning vector.
        if cond_dim != llm_dim:
            self.cond_proj = nn.Sequential(
                nn.Linear(llm_dim, cond_dim),
                nn.SiLU(),
            )
        else:
            self.cond_proj = nn.Identity()

        # One velocity network per segment.
        # For K=1 this is a single global network.
        self.velocity_nets = nn.ModuleList([
            ConsistencyFMVelocityNet(
                action_dim  = action_dim,
                hidden_dim  = hidden_dim,
                cond_dim    = cond_dim,
                n_layers    = n_layers,
            )
            for _ in range(n_segments)
        ])

        # EMA target network (θ-): a deep copy that is updated via EMA,
        # NOT through gradient descent. It provides stable regression targets
        # during consistency loss computation.
        self.target_velocity_nets = copy.deepcopy(self.velocity_nets)
        for p in self.target_velocity_nets.parameters():
            p.requires_grad_(False)   # never receives gradients

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _segment_index(self, t: torch.Tensor) -> torch.Tensor:
        """
        Given timestep t ∈ [0, 1), return which segment index [0, K-1] it
        belongs to.  Segment i covers [i/K, (i+1)/K).
        """
        idx = (t * self.n_segments).long().clamp(0, self.n_segments - 1)
        return idx                                         # [B]

    def _segment_end(self, segment_idx: int) -> float:
        """Return the right endpoint T of segment i."""
        return (segment_idx + 1) / self.n_segments

    def _extract_conditioning(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """
        Extract a single conditioning vector per sample from the LLM's
        full hidden state sequence.

        Strategy: mean-pool over the sequence dimension, then project.
        Alternative: use only the last token hidden state (causal LM
        convention), which is how OFT extracts features for its MLP head.
        Both are tried in practice; mean-pooling is more stable for
        flow-based decoders.

        Args:
            llm_hidden: [B, seq_len, llm_dim]
        Returns:
            c: [B, cond_dim]
        """
        # Mean-pool across sequence length
        c = llm_hidden.mean(dim=1)        # [B, llm_dim]
        c = self.cond_proj(c)             # [B, cond_dim]
        return c

    @torch.no_grad()
    def update_ema(self):
        """
        Update the EMA target network after each gradient step.
        Call this once per training iteration after optimizer.step().
        """
        for online, target in zip(
            self.velocity_nets.parameters(),
            self.target_velocity_nets.parameters(),
        ):
            target.data.mul_(self.ema_decay).add_(
                online.data, alpha=1.0 - self.ema_decay
            )

    # -----------------------------------------------------------------------
    # Velocity prediction (online and target networks)
    # -----------------------------------------------------------------------

    def predict_velocity(
        self,
        x_t:          torch.Tensor,   # [B, action_dim]
        t:            torch.Tensor,   # [B]
        c:            torch.Tensor,   # [B, cond_dim]
        use_target:   bool = False,
    ) -> torch.Tensor:
        """
        Predict velocity using the appropriate segment network.

        Args:
            x_t:        Noisy action at timestep t.
            t:          Flow timestep in [0, 1].
            c:          Conditioned LLM vector (already projected).
            use_target: If True, use the EMA target network (for loss targets).
        Returns:
            v: [B, action_dim]
        """
        nets = self.target_velocity_nets if use_target else self.velocity_nets
        seg_idx = self._segment_index(t)    # [B] — could differ per sample

        # For batches where all samples share the same segment (common during
        # training with uniform-t sampling within a segment), we can call the
        # velocity net once.  For mixed batches, we loop.
        if seg_idx.unique().numel() == 1:
            # Fast path: all samples in the same segment
            s = seg_idx[0].item()
            return nets[s](x_t, t, c)

        # Slow path: samples span multiple segments (rare in practice)
        v = torch.zeros_like(x_t)
        for s in range(self.n_segments):
            mask = seg_idx == s
            if mask.any():
                v[mask] = nets[s](x_t[mask], t[mask], c[mask])
        return v

    # -----------------------------------------------------------------------
    # Consistency loss  (Eq. 6 in the paper, extended for multi-segment)
    # -----------------------------------------------------------------------

    def consistency_loss(
        self,
        x_gt:        torch.Tensor,         # [B, action_dim]  ground-truth action chunk
        llm_hidden:  torch.Tensor,         # [B, seq_len, llm_dim]
        delta_t:     float = 0.05,
    ) -> torch.Tensor:
        """
        Compute the multi-segment velocity consistency loss.

        For each sample:
          1. Sample t uniformly within the current segment.
          2. Construct x_t and x_{t+Δt} by interpolating toward x_gt along
             the OT (optimal transport) straight path:
                 x_t = (1-t)*x_noise + t*x_gt
             where x_noise ~ N(0, I).
          3. Compute f_θ(t, x_t)    = x_t + (1-t)*v_θ(t, x_t)
                     f_θ-(t+Δt, x_{t+Δt}) = x_{t+Δt} + (1-t-Δt)*v_θ-(t+Δt, x_{t+Δt})
          4. Loss = ||f_θ - f_θ-||² + α*||v_θ - v_θ-||²

        Args:
            x_gt:       Ground-truth action (the target x_1 of the flow).
            llm_hidden: Full LLM hidden state sequence for conditioning.
            delta_t:    Small time interval Δt for the consistency target.
        Returns:
            Scalar loss tensor.
        """
        B = x_gt.shape[0]
        device = x_gt.device

        # Extract conditioning vector from LLM hidden states
        c = self._extract_conditioning(llm_hidden)    # [B, cond_dim]

        # Sample noise and construct OT interpolation path
        x_noise = torch.randn_like(x_gt)              # [B, action_dim]

        # Sample t uniformly in [0, 1-Δt] (following the paper)
        t = torch.rand(B, device=device) * (1.0 - delta_t)   # [B]
        t_next = t + delta_t                                  # [B]

        # Straight-line OT interpolation (this is what makes the path straight):
        #   x_t = (1 - t) * x_noise + t * x_gt
        # Under the OT path, x_1 = x_gt with probability 1.
        x_t      = (1 - t.unsqueeze(-1))      * x_noise + t.unsqueeze(-1)      * x_gt
        x_t_next = (1 - t_next.unsqueeze(-1)) * x_noise + t_next.unsqueeze(-1) * x_gt

        # --- Online network predictions ---
        v_online = self.predict_velocity(x_t, t, c, use_target=False)
        # f_θ(t, x_t): "where would we end up if we ran the straight flow to t=1?"
        f_online = x_t + (1 - t.unsqueeze(-1)) * v_online

        # --- Target network predictions (stop-gradient via EMA) ---
        with torch.no_grad():
            v_target = self.predict_velocity(x_t_next, t_next, c, use_target=True)
            f_target = x_t_next + (1 - t_next.unsqueeze(-1)) * v_target

        # --- Consistency loss (Eq. 6) ---
        loss_f = F.mse_loss(f_online, f_target)
        loss_v = F.mse_loss(v_online, v_target)
        loss   = loss_f + self.alpha * loss_v

        return loss

    # -----------------------------------------------------------------------
    # Inference: sample action from noise → x_1
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        llm_hidden:  torch.Tensor,       # [B, seq_len, llm_dim]
        n_steps_per_segment: int = 1,
    ) -> torch.Tensor:
        """
        Sample a continuous action chunk by running the learned straight flow.

        With n_steps_per_segment=1 and K=2 segments this is 2 NFEs total,
        matching the best result in the Consistency-FM paper (FID 5.34, NFE 2).

        For n_steps_per_segment > 1, Euler integration is used within each
        segment (Eq. 13 of the paper), trading more NFEs for higher quality.

        Args:
            llm_hidden:          LLM hidden states for conditioning.
            n_steps_per_segment: Number of Euler steps within each segment.
        Returns:
            x_1: [B, action_dim]  — the predicted action chunk.
        """
        B      = llm_hidden.shape[0]
        device = llm_hidden.device
        c      = self._extract_conditioning(llm_hidden)   # [B, cond_dim]

        # Start from Gaussian noise (the "prior" of the flow)
        x = torch.randn(B, self.action_dim, device=device)

        # March through each segment
        for seg_idx in range(self.n_segments):
            t_start = seg_idx       / self.n_segments
            t_end   = (seg_idx + 1) / self.n_segments
            dt      = (t_end - t_start) / n_steps_per_segment

            for step in range(n_steps_per_segment):
                t_now = t_start + step * dt
                t_vec = torch.full((B,), t_now, device=device)

                # One Euler step: x_{t+dt} = x_t + dt * v_θ^i(t, x_t)
                v = self.velocity_nets[seg_idx](x, t_vec, c)
                x = x + dt * v

        return x   # [B, action_dim]  — denormalize before sending to robot


# ---------------------------------------------------------------------------
# 5.  Integration shim: wraps a frozen OpenVLA-OFT LLM and decoder together
# ---------------------------------------------------------------------------

class OpenVLAWithConsistencyFM(nn.Module):
    """
    Thin integration wrapper showing how ConsistencyFMActionDecoder slots
    into the OpenVLA-OFT stack.

    In a real implementation the LLM backbone (llm) would be the existing
    Llama 2 7B module from OpenVLA-OFT, loaded with pretrained weights and
    fine-tuned via LoRA.  The ConsistencyFMActionDecoder replaces the
    original 4-layer MLP head.

    Args:
        llm:             The Llama 2 backbone (or any module returning a
                         last_hidden_state tensor).
        action_decoder:  The ConsistencyFMActionDecoder defined above.
    """

    def __init__(
        self,
        llm:             nn.Module,
        action_decoder:  ConsistencyFMActionDecoder,
    ):
        super().__init__()
        self.llm            = llm
        self.action_decoder = action_decoder

    def forward(
        self,
        pixel_values:     torch.Tensor,   # [B, C, H, W]  processed by vision stack
        input_ids:        torch.Tensor,   # [B, seq_len]  tokenized instruction
        attention_mask:   torch.Tensor,   # [B, seq_len]
        action_gt:        Optional[torch.Tensor] = None,  # [B, action_dim] for training
    ) -> dict:
        """
        Forward pass for training or inference.

        During training (action_gt provided): returns a loss dict.
        During inference: returns sampled action chunks.

        NOTE: pixel_values are processed by the SigLIP/DINOv2 vision stack
        and projected into the LLM's embedding space before this call.
        That part of the pipeline is unchanged from OpenVLA-OFT.
        """
        # --- LLM forward pass ---
        # In OpenVLA-OFT the vision tokens and language tokens are concatenated
        # before being passed to Llama 2. We assume that has already happened
        # and input_ids / pixel_values arrive as a combined sequence.
        llm_out = self.llm(
            input_ids     = input_ids,
            attention_mask= attention_mask,
            output_hidden_states = True,
        )
        # Take the final layer's hidden states as the conditioning signal.
        # Shape: [B, seq_len, llm_dim]
        llm_hidden = llm_out.hidden_states[-1]

        if action_gt is not None:
            # --- Training path ---
            loss = self.action_decoder.consistency_loss(
                x_gt       = action_gt,
                llm_hidden = llm_hidden,
            )
            return {"loss": loss}
        else:
            # --- Inference path ---
            action_pred = self.action_decoder.sample(
                llm_hidden          = llm_hidden,
                n_steps_per_segment = 1,    # 1 NFE per segment → 2 NFEs total (K=2)
            )
            return {"action": action_pred}


# ---------------------------------------------------------------------------
# 6.  Training loop excerpt
# ---------------------------------------------------------------------------

def training_step_example(
    model:     OpenVLAWithConsistencyFM,
    optimizer: torch.optim.Optimizer,
    batch:     dict,
) -> float:
    """
    Minimal training step showing how EMA update integrates with the
    standard PyTorch optimizer loop.

    Args:
        model:     The full OpenVLA + Consistency-FM model.
        optimizer: AdamW (or similar) operating on trainable params only.
                   In practice: LoRA params in the LLM + all decoder params.
        batch:     Dict with keys pixel_values, input_ids, attention_mask,
                   action (ground-truth action chunk).
    Returns:
        Scalar loss value.
    """
    model.train()
    optimizer.zero_grad()

    out = model(
        pixel_values   = batch["pixel_values"],
        input_ids      = batch["input_ids"],
        attention_mask = batch["attention_mask"],
        action_gt      = batch["action"],
    )

    loss = out["loss"]
    loss.backward()

    # Gradient clipping — important for flow models during early training
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Update EMA target network AFTER the gradient step
    # This must happen every iteration for the consistency loss to be stable.
    model.action_decoder.update_ema()

    return loss.item()


# ---------------------------------------------------------------------------
# 7.  Quick dimensionality smoke-test (runs without a real LLM)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Instantiate the decoder standalone and verify tensor shapes are correct
    for a typical OpenVLA-OFT setup:
        - Action dim: 7 (6-DoF arm + gripper) × chunk_size 8 = 56
        - LLM dim: 4096 (Llama 2 7B)
        - Batch size: 4
    """
    torch.manual_seed(0)

    BATCH       = 4
    SEQ_LEN     = 32     # typical token sequence length after vision tokens
    LLM_DIM     = 4096
    ACTION_DIM  = 56     # 7 DoF × 8-step action chunk

    decoder = ConsistencyFMActionDecoder(
        action_dim  = ACTION_DIM,
        llm_dim     = LLM_DIM,
        hidden_dim  = 512,
        cond_dim    = 256,
        n_segments  = 2,
        n_layers    = 4,
        ema_decay   = 0.999,
        alpha       = 1.0,
    )

    # Fake LLM hidden states and ground-truth actions
    llm_hidden = torch.randn(BATCH, SEQ_LEN, LLM_DIM)
    x_gt       = torch.randn(BATCH, ACTION_DIM)

    # --- Training: loss should be a positive scalar ---
    loss = decoder.consistency_loss(x_gt, llm_hidden)
    print(f"Training loss (should be > 0): {loss.item():.4f}")

    # --- Inference: output shape should match action_dim ---
    action = decoder.sample(llm_hidden, n_steps_per_segment=1)
    print(f"Sampled action shape: {action.shape}")   # expect [4, 56]
    assert action.shape == (BATCH, ACTION_DIM), "Shape mismatch!"

    print("Smoke test passed.")