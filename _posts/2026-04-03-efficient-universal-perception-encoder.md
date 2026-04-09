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


## Research Proposal

- **Swap the backbone in our RAP setup**: RAP currently uses DINOv3-H as its vision encoder. It would be straightforward to substitute the universal encoder here and re-run the NAVSIM benchmark — clean apples-to-apples comparison.
- **Distillation into a lightweight variant**: The full model is 72M params, which is reasonable but not edge-friendly. TinyViT-style distillation from this encoder into a smaller student could be worth doing.
- **Cross-benchmark evaluation**: Worth checking how the encoder holds up on Bench2Drive's closed-loop eval, where sensor noise and distribution shift matter more. Is this model resilient to augmentations or common driving scenarios?

Don't take my word for it — [read their work](https://arxiv.org/pdf/2512.13684).
