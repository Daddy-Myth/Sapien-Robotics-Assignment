## Sapien-Robotics-Assignment Overview

This repository contains solutions to three technical assignments, each implemented and documented in its respective folder.  
The READMEs inside each folder provide detailed explanations of the approach, design choices, and results.

## Question 1 – Faster R-CNN From Scratch for Object Detection

This question focuses on building a **Faster R-CNN–style two-stage object detector entirely from scratch**, without using any pretrained weights. The goal is to develop a clear, end-to-end understanding of modern object detection rather than maximizing benchmark performance.

A filtered subset of the **PASCAL VOC 2012** dataset is used, limited to five classes (*person, car, cat, aeroplane, bicycle*). The model consists of a **custom CNN backbone**, a **Region Proposal Network (RPN)**, and an **ROI-based detection head**, trained jointly using a multi-task loss.

This work highlights key challenges in two-stage detectors, including **training stability, proposal quality, and accuracy–speed trade-offs**, while demonstrating that Faster R-CNN remains a strong and interpretable baseline even when trained from scratch.

## Question 2 – Automated PCB Defect Detection System

This question presents an **automated visual inspection system for Printed Circuit Boards (PCBs)** using deep learning–based object detection. The goal is to identify manufacturing defects **before packaging**, while providing precise localization, defect classification, confidence scores, and severity estimates suitable for industrial use.

The system is built using **YOLOv8**, fine-tuned on a PCB defect dataset containing six defect types and defect-free samples to prevent false positives. It performs real-time inference, outputs bounding boxes and defect centers, and estimates defect severity based on defect area relative to image size.

The project emphasizes **industrial applicability**, demonstrating high accuracy after fine-tuning, real-time performance on GPU, and explicit failure analysis—showing the importance of domain-specific training for reliable manufacturing inspection systems.


## Question 3 – Custom Vision–Language Model for Offline PCB Inspection

This question proposes a **domain-specific vision–language model (VLM)** designed for **offline PCB quality inspection** under strict industrial constraints. Instead of using generic, fully generative VLMs, the system prioritizes **spatial grounding, low hallucination, deterministic outputs, and sub-2-second inference**.

The architecture is inspired by **BLIP-2**, but customized with **detection-aware vision encoders**, region-level visual tokens, and **constrained language decoding**. It leverages existing bounding-box annotations to generate synthetic QA data, avoiding the need for manual text supervision. Structured outputs (defect type, location, confidence) and explicit abstention mechanisms further reduce hallucination risk.

Overall, this design demonstrates how a **modular, grounded, and optimized VLM** can meet real-world industrial inspection requirements where reliability and interpretability matter more than open-ended reasoning.
