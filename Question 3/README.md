# A Custom Vision–Language Model for Offline PCB Quality Inspection

## Abstract

Printed Circuit Board (PCB) inspection in semiconductor manufacturing demands highly reliable visual reasoning under strict latency and deployment constraints. Generic vision–language models (VLMs), while powerful, are unsuitable for this setting due to hallucinations, weak spatial grounding, and high inference costs. This document presents the design of a custom VLM architecture tailored for offline industrial quality inspection, where inspectors query defects using natural language and receive structured, localized, and confidence-calibrated responses within sub-2-second inference time. Leveraging existing bounding-box annotations without manual QA labels, we propose a modular VLM built around detection-aware vision encoders, constrained language decoding, and multi-stage alignment training to minimize hallucinations while maintaining high localization accuracy.


## 1. Introduction

In high-volume semiconductor manufacturing, PCB inspection systems must operate with extreme precision and reliability. Even minor hallucinations or localization errors can lead to costly downstream failures. While modern VLMs such as LLaVA, BLIP-2, and Qwen-VL demonstrate impressive general-purpose reasoning, they are primarily optimized for open-world visual understanding and conversational generation. These properties are misaligned with industrial requirements, where answers must be deterministic, verifiable, and grounded in measurable visual evidence.

The problem addressed in this work is the design of a domain-specific VLM that can answer natural language queries about PCB defects using only image-level supervision in the form of bounding boxes. The system must run fully offline, avoid hallucinations, and return structured outputs including defect type, spatial location, and confidence scores.


## 2. Problem Setting and Constraints

The available dataset consists of approximately 50,000 PCB images annotated with defect bounding boxes. No question–answer pairs or textual descriptions are provided. Inspectors are expected to query the system using natural language (e.g., “Is there a short circuit near the capacitor?”), and the system must return grounded responses with explicit spatial coordinates.

Key constraints shape the system design. First, inference latency must remain below two seconds, ruling out large end-to-end generative VLMs. Second, the system must operate offline, excluding reliance on external APIs or large-scale cloud models. Finally, hallucination is unacceptable: the model must either provide visually grounded evidence or explicitly abstain.

## 3. Model Selection Rationale

The selection of the model architecture is driven by the mismatch between generic vision–language models and the requirements of industrial PCB inspection. While most VLMs emphasize open-ended reasoning and conversational fluency, this task demands precise spatial grounding, deterministic outputs, low hallucination rates, and sub-2-second offline inference. As a result, architectural controllability and deployment constraints take precedence over general benchmark performance.

LLaVA-style architectures tightly couple a global image embedding with a large language model, enabling strong descriptive reasoning but offering limited guarantees of spatial alignment between generated text and specific image regions. This coupling leads to large model sizes, higher inference latency, and increased hallucination risk, making such models unsuitable for safety-critical offline deployment. Qwen-VL improves multimodal reasoning but remains fundamentally generative, with weak output constraints and a heavy deployment footprint that complicates reliable edge inference.

In contrast, BLIP-2-style architectures decouple the vision encoder from the language model, allowing independent optimization of visual grounding and linguistic reasoning. This modular design enables the vision encoder to be trained using bounding-box supervision while constraining the language decoder to operate over region-level visual tokens. Such separation is critical for enforcing spatial grounding, reducing hallucination, and meeting strict latency requirements.

Based on these considerations, we adopt a customized BLIP-2–inspired architecture, extending it with detection-aware visual features and structured decoding to satisfy the reliability and performance demands of industrial PCB inspection.

| Architecture | Strengths | Limitations in PCB Inspection | Suitability |
|-------------|-----------|-------------------------------|-------------|
| LLaVA | Strong general reasoning,<br>conversational fluency | Weak spatial grounding, large<br>model size, slow inference,<br>high hallucination risk | Low |
| BLIP-2 | Modular design, flexible vision<br>encoder, efficient fine-tuning | Requires customization for<br>fine-grained localization | Medium (Base) |
| Qwen-VL | Advanced multimodal reasoning,<br>strong language understanding | Generative bias, weak output<br>constraints, heavy deployment<br>footprint | Low |
| Custom BLIP-2-based<br>VLM (Proposed) | Precise localization, structured outputs,<br>low latency, offline deployable | Requires careful system<br>design and training | High |


## 4. Architecture Design

### 4.1 Vision Encoder

The vision encoder is designed not as a generic image feature extractor but as a detection-aware module. A CNN or compact Vision Transformer backbone is trained using the available bounding box annotations to produce region-level embeddings corresponding to defect candidates. ROI-based feature extraction ensures that each visual token maps directly to a spatially localized region on the PCB.

This design choice ensures that downstream language reasoning operates over explicitly grounded visual units rather than abstract global embeddings, which are a major source of hallucination in generic VLMs.


### 4.2 Vision–Language Fusion

Fusion is implemented via cross-attention mechanisms that allow language tokens to attend selectively to region-level visual embeddings. Unlike captioning models where the image embedding dominates the interaction, here visual regions act as anchors that constrain the language model’s reasoning space. Each answer token is thus conditioned on a specific visual region, enforcing spatial grounding at the architectural level.

### 4.3 Language Decoder and Output Constraints

The language decoder is intentionally constrained. Instead of free-form text generation, outputs are structured into predefined schemas containing defect type, bounding box coordinates, and confidence scores. This restriction significantly reduces the degrees of freedom available to the model, which in turn limits hallucination and improves interpretability.

The decoder is trained to abstain when no sufficient visual evidence exists, allowing the system to respond with “no defect detected” rather than generating unsupported claims.


## 5. Optimization for Offline, Low-Latency Inference

Meeting the sub-2-second inference requirement necessitates optimization at multiple levels. At the model level, quantization (INT8 or INT4) and structured pruning are applied to both vision and language components. Knowledge distillation transfers reasoning capabilities from a larger teacher model to a compact student suitable for deployment.

During fine-tuning, Low-Rank Adaptation (LoRA) is used to update only a small subset of parameters, reducing memory overhead and training time. For deployment, the model is exported using optimized inference runtimes such as ONNX or TensorRT, ensuring efficient execution on local GPUs or CPUs.


## 6. Hallucination Mitigation Strategies

Hallucination mitigation is treated as a first-class design goal rather than a post hoc fix. Architecturally, hallucination is constrained by grounding every answer in region-level visual tokens and disallowing unconstrained text generation. From a training perspective, synthetic negative QA samples are introduced, explicitly teaching the model when not to answer.

Loss functions penalize predictions that lack spatial alignment or produce confident answers without corresponding visual evidence. Confidence calibration losses further ensure that output scores accurately reflect uncertainty, a critical requirement in industrial decision-making.


## 7. Training Methodology

Training proceeds in multiple stages. First, the vision encoder is trained independently as a defect detector using standard localization metrics such as IoU and mean Average Precision. Next, synthetic QA pairs are automatically generated from bounding box annotations by templating natural language questions tied directly to known defect instances.

In the third stage, the vision–language fusion module and language decoder are trained using these synthetic QA pairs while keeping the vision backbone frozen. Finally, a distillation phase compresses the aligned model into a deployable variant without significant loss in accuracy.


## 8. Evaluation and Validation

Validation focuses on three axes: localization accuracy, reasoning correctness, and hallucination rate. Localization performance is measured using IoU and recall against ground-truth bounding boxes. Counting accuracy is evaluated via absolute count error across defect categories. Hallucination is quantified by measuring false positives on defect-free images and the frequency of answers unsupported by visual evidence.

Latency and memory usage are benchmarked under realistic deployment conditions to ensure compliance with offline industrial constraints.


## 9. Conclusion

This document presents a principled design for a custom VLM tailored to PCB quality inspection. By prioritizing modularity, spatial grounding, and constrained generation, the proposed system overcomes the limitations of generic VLMs and aligns with real-world industrial requirements. The resulting architecture achieves reliable localization, low hallucination rates, and fast offline inference, making it suitable for deployment in semiconductor manufacturing environments.
