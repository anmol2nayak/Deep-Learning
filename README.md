🌿 ConvNeXt-Tiny for Advanced Cotton Disease Detection and Severity Estimation

Project Overview

This repository documents a successful Deep Learning comparative study aimed at developing a highly accurate system for classifying major cotton crop diseases. We benchmarked three state-of-the-art CNN architectures—ConvNeXt-Tiny, RegNet, and Xception—and selected ConvNeXt-Tiny as the most robust model for disease diagnosis.

Our work serves as a powerful foundation for a future Precision Agriculture tool focused on quantifiable, actionable data for farmers.

Key Disease Classes

Healthy

Bacterial Blight

Curl Virus

Fusarium Wilt

🚀 Core Achievements & Future Trajectory

Comparative Performance Analysis

We achieved exceptional classification performance by leveraging Transfer Learning on large pre-trained models:

Model

Test Accuracy

Weighted F1-Score

Status

ConvNeXt-Tiny (Best)

1.00

1.00

Selected Backbone

RegNet-Y

0.92

0.92

Comparative

Xception

0.89

0.89

Comparative

Next-Generation Feature: Disease Severity Quantification (The Unique Contribution)

To move beyond simple classification (i.e., answering "What disease?"), our project is structured to implement Image Segmentation (answering "How much damage?"). This is our key contribution to precision agriculture:

Disease Localization: The model will precisely localize the diseased lesions on the cotton leaf.

Severity Index: It will quantify the damage by calculating the exact percentage of the leaf area affected.

Actionable Data: This severity index enables farmers to administer precision chemical applications, reducing waste, cost, and environmental impact.

💻 Repository Setup & Experiments

Prerequisites

Ensure you have Python 3.9+ and install the necessary dependencies using the provided file:

pip install -r requirements.txt


(Note: PyTorch is used for ConvNeXt/RegNet and TensorFlow is used for Xception.)

Getting Started

Clone the repository:

git clone [YOUR_REPO_URL]
cd [YOUR_REPO_NAME]


Data: Due to size constraints, the dataset is not included. The models rely on a structured directory: data/train/, data/val/, and data/test/, each containing class sub-folders (healthy, bacterial_blight, etc.).

Experiment Files

Notebook

Model

Framework

Purpose

Convnext.ipynb

ConvNeXt-Tiny

PyTorch

Primary Implementation (Best Results)

Regnet.ipynb

RegNet-Y

PyTorch

Comparative Analysis

Xception.ipynb

Xception

TensorFlow/Keras

Comparative Analysis

🛣️ Future Work Roadmap

Our immediate focus is on converting the classification backbone into a full diagnostic tool:

Segmentation Implementation: Integrate a Segmentation Head (e.g., U-Net variant) with the ConvNeXt feature extractor to measure the affected leaf area.

Real-Time Optimization: Optimize the final model into a lightweight format (e.g., TFLite, ONNX) for fast, on-device inference on mobile devices.

Generalization and Robustness: Expand training data using images from diverse field environments, varying lighting conditions, and different growth stages to ensure reliable performance across all real-world scenarios.

🎓 References (Key Research)

Tao et al. (2022). Cotton Disease Detection Based on ConvNeXt and Attention Mechanisms.

Aslam A, et al. (2025). Multi-convolutional neural networks for cotton disease detection using synergistic deep learning paradigm.

Zhang et al. (2023). Optimized YOLOv5 algorithm for small target detection in cotton wilt disease.

Krizhevsky et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks.

Yosinski et al. (2014). How transferable are features in deep neural networks?

A Review: Cotton Leaf Disease Detection (General Survey).

Limitations of DL methods for plant disease detection (General Survey on Challenges).
