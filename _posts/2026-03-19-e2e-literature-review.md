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

Before comparing methods, it's worth understanding what we're measuring — and how different benchmarks answer very different questions. The field has historically conflated three distinct evaluation problems: *how well can the model perceive*, *how well can it predict others*, and *how well can it actually drive*. The benchmarks below each illuminate one corner of that space.

A key structural distinction runs through all of them: **open-loop vs. closed-loop** evaluation. Open-loop methods compare predicted trajectories against logged human trajectories — fast and reproducible, but known to be a poor proxy for driving performance. A model that drifts 0.3 m off the human path may be perfectly safe; one that exactly matches the human trajectory may be about to collide with a vehicle the human saw but the model did not. Closed-loop evaluation executes the model in simulation or the real world and measures actual driving outcomes, but is expensive and harder to reproduce across groups.

---

### nuScenes

**Paper:** Caesar et al., CVPR 2020 ([arXiv:1903.11027](https://arxiv.org/abs/1903.11027))
**Task focus:** 3D detection, tracking, trajectory prediction

nuScenes set the template for modern AV datasets. It covers 1,000 twenty-second scenes (700 train / 150 val / 150 test) collected across Boston and Singapore using a full 360° sensor suite: 6 cameras (front, front-left, front-right, back, back-left, back-right), a 32-beam LiDAR, 5 radars, and GPS/IMU. Keyframes are annotated at 2 Hz, yielding ~40,000 annotated frames across 23 object classes.

The **nuScenes Detection Score (NDS)** is a composite that goes beyond raw accuracy:

```
NDS = (1/10) × [5×mAP + TP_score(ATE) + TP_score(ASE) + TP_score(AOE) + TP_score(AVE) + TP_score(AAE)]
```

This penalizes errors in translation, scale, orientation, velocity, and attribute classification — not just localization. Detection uses center-distance thresholds (0.5–4 m) rather than IoU, which tends to inflate scores relative to KITTI-style evaluation.

For trajectory prediction, the task is to forecast 6-second futures (12 waypoints at 2 Hz) for all agents. The standard metrics are **minADE** (minimum Average Displacement Error over K=5 predictions), **minFDE**, and **MissRate** (fraction of predictions with max pointwise L2 > 2 m).

**What it misses:** nuScenes is a perception and prediction benchmark — it does not evaluate planning. The 32-beam LiDAR is sparse by modern standards, which partly explains why camera-only methods (BEVFormer, DETR3D, SparseBEV) have flourished here. The dataset is also small by modern standards; 1,000 scenes covering ~5.5 hours is orders of magnitude less than Waymo or nuPlan.

---

### Waymo Open Dataset

**Paper:** Sun et al., CVPR 2020 ([arXiv:1912.04838](https://arxiv.org/abs/1912.04838))
**Task focus:** 3D detection, 3D tracking, motion prediction

The Waymo Open Dataset is the most rigorous real-world perception benchmark in the field. The perception dataset covers 1,150 scenes (798 train / 202 val / 150 test), each 20 seconds at 10 Hz, collected in San Francisco, Phoenix, and Mountain View. The sensor rig is significantly more capable than nuScenes: 5 LiDARs (1 mid-range spinning top LiDAR + 4 short-range units) and 5 cameras, fully synchronized and calibrated.

The key metric is **mAPH** — mean Average Precision weighted by heading accuracy. Every detected object must have correct orientation to count; a vehicle detected at the right location but pointing the wrong way is penalized. Results are stratified by difficulty: **LEVEL_1** (≥5 LiDAR points in the bounding box) and **LEVEL_2** (fewer than 5 points, i.e., distant or heavily occluded objects). LEVEL_2 is the headline metric in competitive submissions, and at 50–75 m range, objects may have 1–4 LiDAR points total.

The **Waymo Open Motion Dataset (WOMD)** extends this to motion prediction: 103,354 segments, ~570 hours of unique driving data, with agent trajectories and HD maps. The annual Waymo Challenges at CVPR are widely regarded as the most competitive 3D perception leaderboards in the field.

**What it misses:** Like nuScenes, Waymo is a perception and prediction benchmark — it does not execute a model in a real vehicle or simulation. Access requires a Google account and application approval; no commercial use is permitted and data cannot be redistributed, which limits community tooling. There is no planning evaluation.

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

### nuPlan

**Paper:** Karnchanachari et al., ECCV Workshop 2021 ([arXiv:2106.11810](https://arxiv.org/abs/2106.11810)); full benchmark paper [arXiv:2403.04133](https://arxiv.org/abs/2403.04133)
**Task focus:** Planning only (perception given as ground truth)

nuPlan occupies a different niche: it tests the planning module in isolation. Given ground-truth agent states, HD map, and a navigation goal, the model must output an 8-second future trajectory. Perception is assumed solved. The dataset covers ~1,300 hours across four cities (Las Vegas, Boston, Pittsburgh, Singapore) with a rich scenario taxonomy.

Evaluation runs in three modes of increasing realism: open-loop comparison to logged trajectories, closed-loop with scripted (non-reactive) background agents, and closed-loop with **IDM/MOBIL** reactive agents. The primary metric is the **Closed-Loop Score (CLS)**, a composite of comfort, collision avoidance, time-to-collision, progress, and lane compliance.

The nuPlan 2023 challenge produced a striking result: a well-tuned rule-based baseline (PDM-Closed, scoring ~92–93) outperformed all submitted ML planners, raising questions about whether the benchmark actually tests generalization — or just rewards conservative behavior in a dataset dominated by stopped-at-red-light scenarios (~45% of frames). The **nuPlan-R** extension ([arXiv:2511.10403](https://arxiv.org/abs/2511.10403)) replaces IDM with learned diffusion-based reactive agents to increase the challenge.

**What it misses:** nuPlan evaluates planning in isolation — sensor noise, occlusion, and detection failures are all outside its scope. IDM background agents are simplified and don't capture real traffic dynamics. And since perception is given, nuPlan cannot measure the compounding errors that make real E2E driving hard.

---

### What the Benchmarks Collectively Miss

No single benchmark captures the full picture:

- **Real-world closed-loop evaluation at scale** doesn't exist as a public benchmark. Waymo and nuScenes use real data but evaluate offline. CARLA and Bench2Drive close the loop but run in simulation.
- **Sensor-realistic closed-loop evaluation** is largely absent — benchmarks either use real sensors with open-loop evaluation, or simulated sensors with closed-loop evaluation, but rarely both.
- **Long-tail and out-of-distribution scenarios** are underrepresented. Bench2Drive's 44 scenario types cover common critical events, but rare naturalistic scenarios (e.g., unusual road markings, construction zones, sensor degradation) are not systematically benchmarked.
- **The open-loop / closed-loop correlation problem** remains open: it is still unclear what open-loop metric, if any, reliably predicts closed-loop performance. Bench2Drive's finding that L2 does not is a starting point, not a resolution.


---

## Chapter 6: Open Problems

<!-- Distribution shift, causal confusion, interpretability, long-tail scenarios -->

---

## Summary

<!-- Key takeaways and how this review motivates the architectural choices in the next post -->
