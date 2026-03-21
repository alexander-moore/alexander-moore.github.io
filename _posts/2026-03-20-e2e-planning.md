---
layout: article
title: "AV Blog 4: End-to-End Trajectory Planning from Vision"
date: 2026-03-20
description: "Designing and ablating a suite of neural planners — from a pure-kinematics MLP baseline through vision-augmented transformer architectures with auxiliary depth supervision."
tags: [autonomous-vehicles, computer-vision, transformers, trajectory-planning, ablation-study]
image:
  path: /images/claude_vision_transformer_planner.svg
---
(Note: Sections of this post, and the architecture figures, were produced by LLMs)

With depth and segmentation extraction covered in the last post, it's time to tackle the core problem: **end-to-end trajectory planning**. Given sensor history, where should the car go next?

This post introduces six model architectures of increasing complexity, and lays out the ablation study designed to understand which components of a vision-augmented planner actually matter.

Here's a mid-training snapshot from TensorBoard:

<figure>
  <img src="/images/tb_train_loss.png" alt="TensorBoard training loss curves for all six models">
  <figcaption>Training loss (step) across all six models.</figcaption>
</figure>

<figure>
  <img src="/images/tb_val_ade.png" alt="TensorBoard validation ADE curves">
  <figcaption>Validation Average Displacement Error (ADE). Vision-augmented models separate from the kinematics-only baselines (MLP, Transformer) as training progresses.</figcaption>
</figure>

<figure>
  <img src="/images/tb_val_avg_l2.png" alt="TensorBoard validation average L2 curves">
  <figcaption>Validation average L2 displacement error. The Resnet+transformer model is currently leading in this metric. The [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) authors warn against evaluating models in these open-loop metrics as they may not correlate to real-world driving performance.</figcaption>
</figure>

---

## Chapter 1: The Task
<figure>
  <img src="/images/vision_former_traj.png" alt="MLP Planner architecture diagram">
  <figcaption>Figure 1: End-to-end training estimates the future trajectory from the past trajectory and kinematics. In this example, Architecture 4 (front camera planner) predicts a stop at the end of the 5-second sequence - but stops too early.</figcaption>
</figure>

The planner takes in a window of recent sensor data and must predict the **next 50 future waypoints** of the ego vehicle. Inputs available at each timestep:

| Input | Shape | Description |
|---|---|---|
| `past_traj` | (B, 41, 2) | Last ~4s of ego (x, y) positions |
| `speed` | (B, 41) | Ego speed per timestep |
| `acceleration` | (B, 41, 3) | 3-axis acceleration per timestep |
| `command` | (B,) | High-level driving command (turn left/right, go straight, follow lane) |
| `images` | (B, C, 3, 224, 224) | Onboard camera frames — 1 (front) or 6 (surround) |

The planner outputs a future trajectory: `(B, 50, 2)` waypoints over the next ~5 seconds.

> **Why 41 past steps and 50 future steps?** The Bench2Drive dataset runs at ~10Hz. 41 frames ≈ 4 seconds of history; 50 frames ≈ 5 seconds of future — long enough to capture a full lane change or intersection crossing.

---

## Chapter 2: The Architectures

Six architectures are implemented, progressively adding components. Each builds intuition for what contributes to planning performance.

---

### Architecture 1: MLP Planner

The simplest possible baseline. All kinematic inputs are flattened into a single vector and fed through a deep MLP.

<figure>
  <img src="/images/claude_mlp_planner_horizontal.svg" alt="MLP Planner architecture diagram">
  <figcaption>Figure 2: The MLP Planner flattens all kinematic inputs — trajectory, speed, acceleration, and one-hot command — into a single 286-dim vector and passes it through 4 hidden layers to predict 50 future waypoints.</figcaption>
</figure>

**Key design choices:**
- Input: flatten `past_traj` (82) + `speed` (41) + `acceleration` (123) + one-hot `command` (4) = **286-dim** vector
- 4× [Linear(256) → LayerNorm → GELU → Dropout(0.1)]
- Output: Linear(100) reshaped to (50, 2) waypoints
- **Parameters: 289K**

This model is **vision-free**. It asks: *how well can we plan from kinematics alone?* It also gives us a compute and accuracy floor for every subsequent architecture.

---

### Architecture 2: Transformer Planner

The same kinematic inputs — but rather than flattening, we preserve the **temporal structure** of the 41-step history. Each timestep becomes a token; a transformer encoder learns temporal relationships, and a decoder with learned query embeddings generates the future trajectory.

<figure>
  <img src="/images/claude_transformer_planner_horizontal.svg" alt="Transformer Planner architecture diagram">
  <figcaption>Figure 3: The Transformer Planner preserves the temporal axis of kinematic data. Each past timestep is encoded as a 10-dim token (x, y, speed, acc_xyz, command), projected to d=128, and processed by a 3-layer transformer encoder. A 3-layer decoder with 50 learned query embeddings auto-regressively attends to the encoded history.</figcaption>
</figure>

**Key design choices:**
- Per-timestep token: `[x, y, speed, acc_x, acc_y, acc_z, cmd_0..3]` = **10-dim**
- Input projection: Linear(10 → 128) + sinusoidal positional encoding
- 3× TransformerEncoderLayer (4 heads, FFN=512)
- 3× TransformerDecoderLayer with 50 learned query embeddings
- Output: Linear(128 → 2) per query
- **Parameters: 1.4M**

**Ablation question:** Does temporal attention over kinematics outperform the MLP's flattened view?

---

### Architecture 3: ResNet Planner

Now we add vision. The front camera is encoded with a **frozen, pretrained ResNet50** backbone. Visual features are projected to token embeddings and fused with kinematic memory inside a cross-attention decoder.

<figure>
  <img src="/images/claude_resnet_planner_horizontal_v2.svg" alt="ResNet Planner architecture diagram">
  <figcaption>Figure 4: The ResNet Planner uses a frozen ResNet50 to extract visual features from the front camera. Feature maps at 7×7 (and optionally 56×56 through 7×7 multiscale) are projected to d=128 tokens. A kinematic encoder processes trajectory history, and a FlexDecoder cross-attends to both visual and kinematic sources to predict the trajectory.</figcaption>
</figure>

**Key design choices:**
- **Frozen** ResNet50: no gradient flows into the backbone — visual features are treated as a fixed prior
- Supports **single-scale** (7×7 = 49 tokens from layer4) or **multiscale** (56²+28²+14²+7² = 4,410 tokens)
- `KinematicEncoder`: 2-layer transformer encoder over the kinematic sequence
- `FlexDecoderLayer`: self-attention on future queries + cross-attention to [visual tokens | kinematic tokens]
- **Parameters: 33.2M** (dominated by frozen ResNet50 ~25M)

**Ablation questions:**
- Does vision from ResNet50 improve over kinematics-only?
- Does multiscale visual context (small objects at higher resolution) improve planning?

---

### Architecture 4: Front Camera Planner

Swaps the ResNet50 backbone for a **frozen TinyViT** — a lightweight Vision Transformer that generates richer spatial tokens with less compute than a full ViT. Still limited to the **single front camera**.

<figure>
  <img src="/images/claude_front_cam_planner_horizontal.svg" alt="Front Camera Planner architecture diagram">
  <figcaption>Figure 5: The Front Camera Planner uses TinyViT as the visual backbone (frozen). Multiscale visual tokens are combined with kinematic encoder output and decoded by a FlexDecoder to predict 50 future waypoints. 2D sincos positional encodings are used for visual tokens; 1D sincos for kinematics.</figcaption>
</figure>

**Key design choices:**
- **Frozen TinyViT**: produces richer semantic tokens than ResNet's convolutional features
- 2D sinusoidal positional encodings for spatial awareness in the visual tokens
- Otherwise structurally identical to ResNetPlanner
- **Parameters: 29.6M** (dominated by frozen TinyViT ~28M)

**Ablation question:** Does TinyViT's attention-based feature extraction improve over ResNet50's convolution-based features for trajectory planning?

---

### Architecture 5: Front Camera + Depth Planner

Adds an **auxiliary depth estimation head** to the Front Camera Planner. The hypothesis: training to predict depth forces the visual encoder tokens to encode geometry, which then benefits trajectory planning.

<figure>
  <img src="/images/claude_front_cam_depth_planner_v2.svg" alt="Front Camera Depth Planner architecture diagram">
  <figcaption>Figure 6: The Front Camera Depth Planner adds a separate depth decoder head alongside the trajectory decoder. Both share the same frozen TinyViT visual tokens and kinematic encoder. Depth supervision is applied when ground-truth depth maps are available in the batch.</figcaption>
</figure>

**Key design choices:**
- Trajectory decoder and depth decoder are **separate** (no shared weights beyond the backbone tokens)
- Depth head outputs `(B, 1, 224, 224)` — a full-resolution depth map
- Depth loss is only applied when depth ground truth is available (curriculum-friendly)
- At inference, only the trajectory head is used
- **Parameters: 35.7M** (+6.1M depth decoder over Front Cam Planner)

**Ablation questions:**
- Does geometric auxiliary supervision improve trajectory accuracy?
- Is there a trade-off where depth training hurts or helps the planning objective?

---

### Architecture 6: Vision Transformer Planner (Full Model)

The full architecture: **six surround cameras**, multiscale TinyViT features, kinematic encoder, and optional auxiliary depth and semantic segmentation heads — all unified under a shared decoder framework.

<figure>
  <img src="/images/claude_vision_transformer_planner.svg" alt="Vision Transformer Planner architecture diagram">
  <figcaption>Figure 7: The Vision Transformer Planner processes all 6 onboard cameras with a shared frozen TinyViT backbone. Multiscale features from all cameras are pooled and combined with kinematic encoding. Separate decoders predict the future trajectory (primary), depth map, and semantic segmentation (auxiliary).</figcaption>
</figure>

**Key design choices:**
- **6 cameras**: front, front-left, front-right, rear, rear-left, rear-right
- Shared TinyViT backbone across all cameras (weight sharing)
- Multiscale tokens: 4 levels (56×56, 28×28, 14×14, 7×7) from all cameras
- 3 decoder heads: trajectory, depth, segmentation
- 2D sincos pos enc applied per-camera, 1D sincos for kinematics
- Auxiliary heads detached from trajectory decoder (no gradient leakage)

**Ablation questions:**
- Does surround-view context (6 cameras vs 1) improve planning?
- Does adding semantic segmentation as an auxiliary task help further?

---

## Chapter 3: Ablation Study Design

The six architectures are not arbitrary — they form a controlled ablation ladder across three independent axes.

### Axis 1: Visual Input

| Model | Vision | Cameras | Parameters | Latency (ms) | FPS | Avg L2 (↓) |
|---|---|---|---|---|---|---|
| MLP Planner | None | — | 289K | 0.4 | 2550 | — |
| Transformer Planner | None | — | 1.4M | 2.3 | 440 | — |
| ResNet Planner | ResNet50 (frozen) | 1 (front) | 33.2M | 7.7 | 130 | — |
| Front Cam Planner | TinyViT (frozen) | 1 (front) | 29.6M | 8.8 | 113 | — |
| FrontCam+Depth Planner | TinyViT (frozen) | 1 (front) | 35.7M | 11.9 | 84 | — |
| ViT Planner (full) | TinyViT (frozen) | 6 (surround) | 41.8M | 66.3 | 15 | — |
| *ThinkTwice (SOTA)* | — | — | — | — | — | *0.95* |

*Avg L2: mean L2 distance (meters) between predicted and ground-truth waypoints on the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) validation split. Lower is better. SOTA is 0.95m from ThinkTwice. Results populated as training completes. Latency measured at batch size 1 on an RTX 3090 (fp32); Bench2Drive runs at 10 Hz so ≥10 FPS is required for real-time inference.*

**Controls:** does vision help planning? Does more coverage (surround vs front only) help?

---

### Axis 2: Backbone Architecture

| Backbone | Type | Parameters | Frozen? |
|---|---|---|---|
| None (MLP/Transformer) | — | — | — |
| ResNet50 | CNN | ~25M | Yes |
| TinyViT | ViT | ~28M | Yes |

**Control:** CNN features vs. attention-based features — which provides better planning cues?

---

### Axis 3: Auxiliary Supervision

| Model | Depth Head | Segmentation Head |
|---|---|---|
| Front Cam Planner | No | No |
| FrontCam+Depth Planner | Yes | No |
| ViT Planner (full) | Yes | Yes |

**Control:** does geometric/semantic auxiliary supervision act as a regularizer that improves trajectory prediction?

---

### Metrics

All models are evaluated on the **Bench2Drive validation split** using:

- **L2 displacement error** at 1s, 2s, 3s, 5s horizons (meters)
- **Final Displacement Error (FDE)** at 5s horizon
- **Collision rate** (% of rollouts with ego-obstacle contact, when available)

Auxiliary task metrics (depth AbsRel, segmentation weighted IoU) are tracked separately to monitor auxiliary head quality and its correlation with trajectory performance.

---

### Hypotheses

1. **Vision helps** — even a single front camera should give meaningful improvement over kinematics-only planners, especially in intersection and lane-change scenarios.
2. **TinyViT > ResNet50** for this task — attention-based spatial features should encode scene geometry better than convolutional features.
3. **Depth auxiliary supervision helps** — geometric supervision should force the shared tokens to encode 3D structure, improving obstacle avoidance.
4. **Surround cameras help at longer horizons** — at 1s horizon, forward camera may be enough; at 5s, knowing what's behind and to the side matters.
5. **Segmentation auxiliary has diminishing returns** — semantic labels are noisier than depth for trajectory tasks; this one we're less sure about.

---

## Chapter 4: What's Next

With the architectures designed and the ablation storyboarded, the next steps are:

1. **Train all six models** to convergence under identical hyperparameters
2. **Evaluate on the validation set** across all displacement horizons
3. **Visualize planned trajectories** on held-out scenarios — especially failure cases
4. **Quantify the contribution** of each ablation axis in an aggregated table

The goal is not just a best-performing model, but a **clear understanding of which components earn their compute**.

Follow along with the code:
- [E2E Driving (GitHub)](https://github.com/alexander-moore/drive_e2e)
