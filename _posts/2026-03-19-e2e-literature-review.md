---
layout: article
title: "AV Blog 3: E2E Task - Literature Review"
date: 2026-03-19
description: "A survey of key papers in end-to-end autonomous driving — covering foundational approaches, recent transformer-based planners, and the open problems that motivate this work."
tags: [autonomous-vehicles, computer-vision, literature-review, transformers, trajectory-planning]
---

Before designing a planner, it helps to understand what has already been tried. This post surveys the key literature in end-to-end autonomous driving — the papers that define the problem, establish baselines, and point toward what remains unsolved.

---

## Chapter 1: What Is End-to-End Driving?

<!-- Define the E2E paradigm vs. modular pipelines. What are the promises and tradeoffs? -->
End-to-end driving is the control of a vehicle via an input stream of sensors to predict the desired path of the vehicle as a sequence of future waypoints.

### Important Metrics:
Losses are usually calculated as the L1 distance between the ground-truth trajectory (desired waypoint sequence) against the predicted waypoint trajectory conditioning on:
- historic waypoint sequence (41, 2) x,y points relative to current-frame ego position
- kinematics acceleration (41, 3) 3d acceleration m/s/s and speed (41,) m/s
- driving command (Left, Right, Straight, Lanefollow)

---

## Chapter 2: Foundational Works

<!-- Early imitation learning and behavioural cloning approaches -->
[RAP 3D Rasterization-Augmented E2E Planning]([url](https://arxiv.org/pdf/2510.04333)) ([project page]([url](https://alan-lanfeng.github.io/RAP/))) Proposes that end-to-end driving and the understanding of objects in complex scenes is dependent on planning and object motion dynamics, not on high-resoltuion graphics. To this end, they introduce an image encoder and lightweight simulation space such that the latent representation of the fast-rendering simulation space match corresponding scenes of natural images. This mirrored-latent approach means driving scenarios can be simuilated much faster than when rendering HD graphics - leading to models which understand driving in a real 3d world.
- SOTA on Bench2Drive, SOTA Waymo Open Vision-Based E2E

[DriveTransformer]([url](https://openreview.net/pdf?id=M42KR4W9P5)) proposes a single shared model for end-to-end detection, prediction, mapping and planning from visual and kinematics inputs. Cross-attention between modalities provides a multimodal approach to scene understanding without hand-crafted heuristics or supervised intermediate representations.

[SwinTrajectory]([url](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%203rd%20Place%20-%20Swin-Trajectory.pdf)) 

[SwinTrack]([url](https://arxiv.org/pdf/2112.00995))

---

## Chapter 3: Vision-Based Planners

<!-- Camera-only and multi-camera methods; how visual representations are learned -->

---

## Chapter 4: Transformer Architectures for Planning

<!-- Attention-based models, query-based decoders, temporal modeling -->

---

## Chapter 5: Datasets and Benchmarks

This project focuses on **vision-only end-to-end driving** — raw camera images in, trajectory out, no LiDAR. That scope rules out most classic AV benchmarks, which are built around LiDAR perception and open-loop evaluation. The three benchmarks that matter here are CARLA/Bench2Drive (simulation-based, closed-loop) and the Waymo Vision-based E2E Challenge (real-world, open-loop with a novel human-preference metric). nuScenes and nuPlan are worth knowing about, but as the Bench2Drive authors demonstrate, neither is well-suited for evaluating E2E vision planners.

The central methodological fault line is **open-loop vs. closed-loop** evaluation. Open-loop methods compare predicted trajectories against logged human trajectories — fast and reproducible, but a poor proxy for actual driving. The Bench2Drive paper shows this directly: UniAD achieves lower open-loop L2 error than VAD, yet performs *worse* in closed-loop simulation. Closed-loop evaluation, which executes the model in simulation and measures real driving outcomes, is the only reliable signal for whether a planner actually works.

> **nuScenes** (Caesar et al., CVPR 2020, [arXiv:1903.11027](https://arxiv.org/abs/1903.11027)) — 1,000 scenes with cameras, LiDAR, and radar, primarily used for 3D detection and trajectory prediction. Not a planning benchmark; evaluation is open-loop; heavily LiDAR-centric despite the camera community's use of it. Not relevant for vision-only E2E driving evaluation.

> **nuPlan** (Karnchanachari et al., 2021, [arXiv:2106.11810](https://arxiv.org/abs/2106.11810)) — a planning benchmark that provides ground-truth perception as input, bypassing the vision stack entirely. A well-tuned rule-based planner (PDM-Closed) outperformed all ML submissions in the 2023 challenge, suggesting the benchmark doesn't stress the parts of the problem that matter for E2E learning.

---

### CARLA and the CARLA Leaderboard

**Paper:** Dosovitskiy et al., CoRL 2017 ([arXiv:1711.03938](https://arxiv.org/abs/1711.03938))
**Task focus:** Closed-loop end-to-end driving (simulation)

CARLA (Car Learning to Act) is an open-source simulator built on Unreal Engine 4, providing a Python API for sensors, traffic, weather, and map control. It has become the standard environment for closed-loop E2E driving research, largely because it is free, fully configurable, and provides ground-truth labels for everything.

The CARLA Leaderboard formalizes evaluation into comparable routes across different towns:

| Leaderboard | Routes | Towns | Typical route length |
|---|---|---|---|
| LB 1.0 (2020–21) | 36 ("Longest6") | Towns 1–6 | ~1–2 km |
| LB 2.0 (2022–24) | 20 | Town 13 (10×10 km²) | ~12 km |

The primary metric is **Driving Score (DS) = Route Completion (RC) × Infraction Score (IS)**. Infractions — collisions with pedestrians, vehicles, or static objects; red light violations; stop sign violations; route deviations — each apply a multiplicative penalty to IS, so a single pedestrian collision (×0.50) can cut the score in half.

LB 2.0 introduced much longer routes (12 km vs. ~1.5 km) through Town 13, a large heterogeneous map combining urban high-rise areas, rural farmland, and highways. Each route also includes ~90 injected safety-critical scenarios (cut-ins, jaywalkers, debris, adversarial vehicles) to prevent agents from exploiting route-following shortcuts.

**What it misses:** CARLA's UE4 graphics are not photorealistic, making sim-to-real transfer an ongoing open problem. Historically, background agents did not react to the ego vehicle — LB 2.0 addressed this partially with injected adversarial scenarios, but non-adversarial background traffic is still scripted. CARLA Leaderboard scores are also known to be gameable: high Route Completion is achievable with conservative (slow) driving that avoids infractions by simply never encountering them.

---

### Bench2Drive

**Paper:** Jia et al., NeurIPS 2024 ([arXiv:2406.03877](https://arxiv.org/abs/2406.03877))
**Task focus:** Closed-loop E2E driving, disentangled skill evaluation

Bench2Drive is the benchmark directly used in this project. It was built specifically to address two problems with prior E2E evaluation: open-loop metrics don't predict closed-loop performance, and CARLA Leaderboard routes don't disentangle which driving skills a model has or lacks.

The training dataset covers 2 million annotated frames from ~13,638 clips collected by **Think2Drive**, an RL expert model trained with world models. Clips span 44 interactive scenario types, 23 weather conditions, and 12 CARLA towns — uniformly distributed to prevent shortcuts.

Evaluation runs on **220 short routes** (~150 m each), with each route containing exactly one of the 44 scenario types. This gives a per-skill breakdown: you can see directly whether a model handles cut-ins, emergency braking, or detours, rather than getting a single aggregate score that may mask failure modes.

Metrics follow the CARLA Leaderboard formula (DS = RC × IS) with two modifications: the minimum speed penalty is removed (to avoid penalizing cautious agents), and the evaluation window is extended (2,000 → 4,000 ticks per route). A **Success Rate (SR)** metric — fraction of routes completed without any infraction — is also reported.

The central empirical finding of the Bench2Drive paper is pointed: **open-loop L2 error does not predict closed-loop Driving Score**. UniAD achieves lower L2 than VAD on the open-loop validation set, yet scores worse in closed-loop simulation. This is the main reason this project uses Bench2Drive over open-loop benchmarks.

**What it misses:** The dataset is entirely synthetic (CARLA). Short 150 m routes don't test long-horizon consistency. The training distribution reflects Think2Drive's driving style, which may not match what a learned model needs to see. A follow-up, **Bench2Drive-R** ([arXiv:2412.09647](https://arxiv.org/abs/2412.09647)), addresses the real-world gap by converting logged real-world data into reactive closed-loop scenarios via generative models.

---

### Waymo Vision-Based End-to-End Driving Challenge

**Dataset paper:** Xu et al., 2025 ([arXiv:2510.26125](https://arxiv.org/abs/2510.26125))
**Task focus:** Open-loop waypoint prediction from real-world cameras, long-tail scenarios

The **WOD-E2E** dataset is Waymo's first vision-only E2E benchmark, introduced as a CVPR 2025 challenge track. It is the only major real-world camera-only E2E benchmark currently available. The task: given 12 seconds of 8-camera 360° video (10 Hz), ego state history, and a high-level routing command, predict the next 5 seconds of waypoints (at 4 Hz).

The dataset is deliberately focused on **long-tail scenarios** — rare events that occur at less than 0.03% frequency in normal driving logs (construction zones during events, pedestrians falling, unexpected freeway obstacles). This makes it harder and more safety-relevant than benchmarks sampled uniformly from routine driving.

| Split | Segments | Duration |
|---|---|---|
| Train | 2,037 | ~6.8 hr |
| Val | 479 | ~1.6 hr |
| Test (held-out) | 1,505 | ~5.0 hr |

The standout methodological contribution is the **Rater Feedback Score (RFS)** — a human-preference metric designed to replace geometric distance metrics in long-tail settings. Three human raters independently score each predicted trajectory on safety, legality, reaction time, braking necessity, and efficiency (0–10 scale). A trust region is defined around the rater's own reference trajectory; predictions within the region receive full score, outside it the score decays exponentially. Final RFS is the max score across raters, averaged over time. The paper shows RFS has much stronger correlation with real driving quality in rare scenarios than ADE does.

The inaugural 2025 challenge drew 19 submissions. A notable result: the **Poutine** team (Mila, Quebec AI Institute) topped the public leaderboard at RFS 7.986 using a two-stage approach — vision-language-trajectory pre-training followed by GRPO reinforcement learning from fewer than 500 human-labeled preference frames. They were granted a Special Mention (ineligible for prizes under competition rules). The prize-eligible winner was **UniPlan** (EPFL, RFS 7.779), a diffusion-based unified planner. Third place went to **Swin-Trajectory** (Hanyang University, RFS 7.543), a lightweight Swin Transformer approach with 36M parameters.

**What it misses:** Evaluation is open-loop — the model is never actually executed in a vehicle or simulation, so collision avoidance and reactive behavior are not directly tested. The RFS metric, while more informative than ADE, still relies on logged trajectories as reference and inherits biases from human rater judgment. The dataset is also small (~12 hours total) compared to large-scale simulation benchmarks.

---

### What These Benchmarks Collectively Miss

Even within the closed-loop vision-only scope, gaps remain:

- **Sim-to-real transfer** is unvalidated. Both CARLA and Bench2Drive use UE4 graphics, which are not photorealistic. Models that score well in simulation may not transfer to real camera images.
- **Short routes** in Bench2Drive (~150 m each) don't test long-horizon consistency — a model that handles each scenario in isolation may still fail on a multi-minute real drive.
- **Non-reactive background traffic** means agents can exploit scripted behavior patterns rather than learning to respond to genuine interaction.
- **The open-loop / closed-loop correlation problem** is still open: no open-loop metric reliably predicts closed-loop performance. Bench2Drive establishes that L2 doesn't — but doesn't yet tell us what does.


---

## Chapter 6: Open Problems

<!-- Distribution shift, causal confusion, interpretability, long-tail scenarios -->

---

## Summary

<!-- Key takeaways and how this review motivates the architectural choices in the next post -->
