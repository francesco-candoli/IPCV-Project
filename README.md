# IPCV-Project
A two-part Computer Vision project implementing book recognition using classical feature matching (SIFT/RANSAC) and pet breed classification using custom Convolutional Neural Networks (CNNs) in PyTorch.
---
Module 1: Product Recognition of Books
A traditional computer vision pipeline designed to identify specific books on a shelf using reference images.
- Technique: Feature-based object detection using SIFT (Scale-Invariant Feature Transform) for keypoint extraction.
- Matching: Implements Brute-Force Matching with Lowe’s Ratio Test to filter high-quality feature matches.
- Localization: Computes Homography matrices using RANSAC to project bounding boxes from model images onto the scene.
- Validation: Applies geometric constraints (solidity, convexity, and aspect ratio) to filter false positives.

Module 2: Pet Classification
A Deep Learning approach to classify 37 breeds of cats and dogs using the Oxford-IIIT-Pet dataset.
- Architecture: Implementation of a custom Convolutional Neural Network (CNN) from scratch using PyTorch, featuring 5 convolutional blocks with Batch Normalization and Dropout.
- Training Pipeline: Includes a custom training loop with Early Stopping, Learning Rate Scheduling (CyclicLR), and data augmentation (random crops, flips, and color jitter).
- Evaluation: Tracks training metrics (Loss/Accuracy) and performs ablation studies on model components.

Technologies & Libraries
- Python
- OpenCV (SIFT, Homography, Image Processing)
- PyTorch & Torchvision (CNN implementation, Datasets, Transforms)
- NumPy, Matplotlib, Pandas (Data manipulation and visualization)
