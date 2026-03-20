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

<!-- nuScenes, CARLA, Bench2Drive, Waymo Open — what they test and what they miss -->


---

## Chapter 6: Open Problems

<!-- Distribution shift, causal confusion, interpretability, long-tail scenarios -->

---

## Summary

<!-- Key takeaways and how this review motivates the architectural choices in the next post -->
