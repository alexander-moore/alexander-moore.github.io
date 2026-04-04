---
layout: article
title: "AV Blog 8: Paper Review: Efficient Universal Perception Encoder"
date: 2026-04-03
description: "Reviewing FAIR EURPE — Efficient Universal Perception Encoder."
tags: [autonomous-vehicles, paper-review, computer-vision, transformers, efficiency]
image:
  path: https://github.com/user-attachments/assets/61be798c-9888-469b-be24-1019378d8f4c

---

(Note: This post was produced WITHOUT LLMs)

[GitHub](https://github.com/facebookresearch/eupe) | [Project Page](https://github.com/facebookresearch/eupe) | [ArXiv](https://arxiv.org/pdf/2603.22387)

Foundation vision encoders can be convolutional neural networks or vision transformers trained by full  supervision (like SAM), weak supervision on image-text pairs (like CLIP), or self-supervision (like MAEs).

A single visual foundation model usually excels in one or two task domains, like image-text alignment or dense prediction (like depth, semantic segmentation...). Downstream applications require careful selection of a specific encoder to avoid performance degredation. FAIR's Perception Encoder (PE, https://arxiv.org/abs/2504.13181) proposes to unify many downstream tasks at various depths in a single vision encoder.

However, the authors wonder about _aggregating multiple domain experts into a single model_.q RADIO distills multiple teacher models, which works well for large students but fails for small efficient models. Efficient, fast models are vital for AI on the edge and real-time applications. Self-driving cars, perhaps!

The research proposes a training recipe for efficient encoders, which is first scaling the models up then down. They first implement a proxy teacher which is a large model to distil multiple-expert information. Then train an efficient student from this proxy teacher.

<img width="1351" height="1063" alt="image" src="https://github.com/user-attachments/assets/06ee5770-7970-4841-ab9c-4d1048330736" />
Caption: SOTA across both spatial, captioning, 

<img width="1335" height="521" alt="image" src="https://github.com/user-attachments/assets/3d9a1ea7-e0fe-4b29-965b-21e7d56b01a6" />
Caption: Multi-stage distillation pipeline first unifies multiple experts in a single heaviweight model before distillation into an efficient student - first at fixed resolution, then multiple resolutions!

<img width="1291" height="478" alt="image" src="https://github.com/user-attachments/assets/186c40f7-eb8b-4b48-95dd-fda2402d06a1" />
Caption: EUPE-ViT-B generally performs across image understanding, VLM OCR, scene knowledge, and dense predicition tasks _simultaneously_

**Research Proposal**: Maybe our RAP 3D image encoder could be another teacher in the recipe which we then distill?

They demonstrate that this pipeline leads to efficient models which can **outperform domain experts** when transfering to downstream tasks - even outperforming DINOv3 on dense prediction downstreams!

Contributions:

1. Scale-up scale-down distillation recipe (new take on agglomerative methods like RADIO).

2. A zoo of efficient models (I'm interested in exploring here, especially for video streaming).

3. Study on distillation recipe for training stages, teachers, hyperparameter choices.




<figure>
  <img src="/images/placeholder_architecture.png" alt="[Placeholder: Model architecture overview]">
  <figcaption>[Placeholder caption]: Overview of the proposed universal perception encoder. Left: multi-sensor input (camera, LiDAR, radar). Center: shared encoder backbone with cross-modal attention. Right: downstream task heads for detection, segmentation, and planning.</figcaption>
</figure>

---

## Chapter 1: Background

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. The core challenge in building a universal perception encoder is handling the heterogeneity of sensor modalities — cameras produce dense RGB tensors, LiDAR produces sparse point clouds, and radar produces even sparser reflectance maps. Prior work has generally handled each modality with a dedicated backbone, which is expensive at inference time.

**Key ideas from this paper:**

- **[Idea 1]**: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
- **[Idea 2]**: Ut enim ad minim veniam, quis nostrud exercitation ullamco.
- **[Idea 3]**: Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.

**Related Work**

Universal perception backbones have gained traction recently. [Citation] introduced a joint camera-LiDAR tokenization scheme, while [Citation] demonstrated that a single ViT trunk could be fine-tuned across perception tasks with minimal task-specific overhead. The paper under review builds on this line of work but introduces [key differentiator — to be filled in].

**Research note**: It would be interesting to ablate the cross-modal attention module against a simple concatenation baseline — does the structured interaction actually help, or is it just more parameters?

---

## Chapter 2: Method

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis molestie dictum semper, libero libero commodo lacus, id tincidunt eros neque eget nibh.

The encoder processes all sensor inputs through a shared tokenizer, producing a unified token sequence that is fed into a standard transformer trunk. The key components are:

1. **[Module 1]**: Lorem ipsum dolor sit amet — produces per-sensor token embeddings at a fixed resolution.
2. **[Module 2]**: Cross-modal attention — allows tokens from different modalities to attend to one another prior to the main trunk.
3. **[Module 3]**: Task-agnostic trunk — a standard ViT-B/16 operating over the combined token sequence.

```python
# Placeholder: encoder forward pass
encoder = UniversalPerceptionEncoder(
    backbone="vit_b_16",
    modalities=["camera", "lidar"],
    cross_modal_layers=4,
)

tokens = encoder(camera=imgs, lidar=pts)
# tokens: (B, N, D) — unified token sequence for downstream heads
```

<figure>
  <img src="/images/placeholder_crossmodal.png" alt="[Placeholder: Cross-modal attention visualization]">
  <figcaption>[Placeholder caption]: Cross-modal attention maps between camera tokens (rows) and LiDAR tokens (columns) on a sample driving scene. Brighter cells indicate higher attention weight. The encoder attends most strongly to LiDAR returns in the immediate foreground of the vehicle.</figcaption>
</figure>

**Research note**: The cross-modal attention mechanism here is similar to the raster-to-real alignment in RAP — both are trying to produce a shared latent between two views of the same scene. Worth comparing how the training objectives differ.

---

## Chapter 3: Experiments

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Proin vel ante a orci tempus eleifend ut et magna. They evaluate on three standard benchmarks: [Benchmark A], [Benchmark B], and nuScenes detection.

| Model | [Metric A] ↑ | [Metric B] ↑ | Latency (ms) | Params |
|---|---|---|---|---|
| [Baseline 1] | 52.3 | 41.7 | 48 | 86M |
| [Baseline 2] | 55.1 | 44.2 | 63 | 120M |
| **[This Paper]** | **58.9** | **47.6** | **31** | **72M** |

The proposed model achieves state-of-the-art on both metrics at roughly half the latency of [Baseline 2], despite fewer parameters. The gains on [Metric B] are especially notable — [to be filled in].

<figure>
  <img src="/images/placeholder_results_chart.png" alt="[Placeholder: Results chart — metric vs latency tradeoff]">
  <figcaption>[Placeholder caption]: Accuracy vs. latency tradeoff across compared models. The proposed encoder (star) sits on the Pareto frontier — matching or exceeding prior methods while operating faster.</figcaption>
</figure>

---

## Chapter 4: Ablations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce et ipsum vitae ipsum tristique lobortis id vitae nunc.

| Ablation | [Metric A] | Notes |
|---|---|---|
| Full model | 58.9 | — |
| w/o cross-modal attention | 55.4 | −3.5 pts |
| w/o [Module 2] | 56.1 | −2.8 pts |
| Camera-only | 49.7 | −9.2 pts |
| LiDAR-only | 51.3 | −7.6 pts |

The ablation confirms that cross-modal attention is the single largest contributor to performance. Removing [Module 2] also hurts, though less severely. Neither modality alone approaches the fused result, which is expected but good to verify empirically.

**Research note**: [To be filled in — any surprising ablation results worth calling out?]

---

## Chapter 5: How It Fits into the E2E Stack

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent dapibus, neque id cursus faucibus, tortor neque egestas augue, eu vulputate magna eros eu erat. A universal encoder that produces a single token sequence is a natural fit upstream of an E2E planner — the planner can treat perception as a black box and just consume tokens, regardless of sensor configuration.

The key open questions for integrating this into our pipeline:

- **Temporal context**: Does the encoder produce per-frame tokens, or does it incorporate a temporal window? If per-frame, we'd need to add a temporal aggregation step (similar to what we're already doing with VideoMAE).
- **Resolution compatibility**: Our current setup uses 224×224 camera inputs. If the encoder's tokenizer expects a different resolution, we'd need to adapt it.
- **Downstream task heads**: The paper evaluates detection and segmentation heads. A planning head is straightforward to add in principle — worth trying against our existing trajectory prediction baseline.

<figure>
  <img src="/images/placeholder_integration.png" alt="[Placeholder: Integration diagram for E2E pipeline]">
  <figcaption>[Placeholder caption]: Proposed integration of the universal perception encoder into our E2E pipeline. The encoder replaces the per-modality backbone stack; its output token sequence feeds directly into the planning transformer.</figcaption>
</figure>

---

## Chapter 6: What's Next

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed lacinia, urna non tincidunt mattis, tortor neque adipiscing diam, a cursus ipsum ante quis turpis. A few directions worth exploring:

- **Swap the backbone in our RAP setup**: RAP currently uses DINOv3-H as its vision encoder. It would be straightforward to substitute the universal encoder here and re-run the NAVSIM benchmark — clean apples-to-apples comparison.
- **Distillation into a lightweight variant**: The full model is 72M params, which is reasonable but not edge-friendly. TinyViT-style distillation from this encoder into a smaller student could be worth doing.
- **Cross-benchmark evaluation**: The paper focuses on [Benchmark A/B]. Worth checking how the encoder holds up on Bench2Drive's closed-loop eval, where sensor noise and distribution shift matter more.

Don't take my word for it — [read their work](#).
