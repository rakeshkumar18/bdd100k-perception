# YOLOv8 Architecture

## Objective

Understand the internal architecture of YOLOv8 and how it performs real-time object detection for autonomous driving datasets such as BDD100K.

---

# Why YOLOv8?

YOLO (You Only Look Once) is a one-stage object detector that predicts object locations and classes in a single forward pass through the network.

YOLOv8 was selected because it provides:

* Strong accuracy-speed tradeoff
* Real-time inference capability
* Simple deployment workflow
* State-of-the-art performance for many object detection tasks
* Excellent support through the Ultralytics ecosystem

For autonomous driving applications, low latency is critical. YOLOv8 offers significantly faster inference compared to traditional two-stage detectors.

---

# High-Level Architecture

YOLOv8 consists of three major components:

```text
Input Image
     │
     ▼
 ┌─────────┐
 │ Backbone│
 └─────────┘
     │
     ▼
 ┌─────────┐
 │  Neck   │
 └─────────┘
     │
     ▼
 ┌─────────┐
 │  Head   │
 └─────────┘
     │
     ▼
Bounding Boxes
Classes
Confidence Scores
```

---

# 1. Backbone

## Purpose

The backbone extracts hierarchical visual features from the input image.

Input:

```text
640 × 640 × 3 RGB Image
```

Output:

```text
Multi-scale Feature Maps
```

The backbone learns increasingly complex visual representations.

### Early Layers

Detect:

* Edges
* Corners
* Lines

### Intermediate Layers

Detect:

* Wheels
* Windows
* Traffic signs
* Traffic lights

### Deep Layers

Detect:

* Cars
* Trucks
* Pedestrians
* Buses

The backbone converts raw pixel information into meaningful feature representations used for object detection.

---

# Why Feature Maps Matter

Different objects appear at different scales.

Examples from BDD100K:

| Object        | Relative Size |
| ------------- | ------------- |
| Traffic Light | Small         |
| Traffic Sign  | Small         |
| Person        | Medium        |
| Car           | Medium        |
| Bus           | Large         |
| Truck         | Large         |

To handle this variation, YOLO generates feature maps at multiple resolutions.

Example:

```text
80 × 80
40 × 40
20 × 20
```

Each resolution specializes in detecting objects of different sizes.

---

# 2. Neck

## Purpose

The neck combines information from multiple feature scales.

YOLOv8 uses a combination of:

* Feature Pyramid Network (FPN)
* Path Aggregation Network (PAN)

---

## Problem

Deep layers contain strong semantic information but poor spatial resolution.

Shallow layers contain strong spatial information but weaker semantics.

Without feature fusion:

* Small object detection suffers.
* Localization quality decreases.

---

## Feature Pyramid Network (FPN)

FPN propagates semantic information downward.

```text
Deep Features
      ↓
Medium Features
      ↓
Shallow Features
```

Benefits:

* Better understanding of small objects
* Improved context awareness

---

## Path Aggregation Network (PAN)

PAN propagates localization information upward.

```text
Shallow Features
      ↑
Medium Features
      ↑
Deep Features
```

Benefits:

* Improved object localization
* Better spatial precision

---

## Result

Each feature map contains:

* Semantic information
* Spatial information

This significantly improves detection performance across object scales.

For BDD100K, this is particularly important for:

* Traffic lights
* Traffic signs
* Pedestrians
* Riders
* Motorcycles

---

# 3. Detection Head

## Purpose

The detection head converts feature maps into object predictions.

For every detected object, YOLO predicts:

* Bounding box coordinates
* Object confidence
* Class probabilities

Output example:

```text
Class: Car
Confidence: 0.95
Bounding Box:
(x1, y1, x2, y2)
```

---

# Anchor-Free Detection

YOLOv8 uses an anchor-free detection strategy.

Previous YOLO versions relied on predefined anchor boxes.

YOLOv8 directly predicts object locations from feature map locations.

Benefits:

* Simpler training
* Fewer hyperparameters
* Better generalization
* Easier optimization

---

# Multi-Scale Detection

Predictions are generated at multiple resolutions.

| Scale             | Target Objects |
| ----------------- | -------------- |
| High Resolution   | Small Objects  |
| Medium Resolution | Medium Objects |
| Low Resolution    | Large Objects  |

Examples for BDD100K:

### High Resolution

* Traffic lights
* Traffic signs

### Medium Resolution

* Persons
* Riders
* Bikes
* Motorcycles

### Low Resolution

* Cars
* Trucks
* Buses

This enables robust detection across a wide range of object sizes.

---

# Non-Maximum Suppression (NMS)

Multiple predictions may correspond to the same object.

Example:

```text
Car 95%
Car 92%
Car 89%
```

Non-Maximum Suppression removes duplicate detections and retains only the highest-confidence prediction.

Result:

```text
Car 95%
```

This reduces duplicate detections and improves output quality.

---

# YOLOv8 Detection Pipeline

```text
Input Image
      │
      ▼
Backbone
      │
      ▼
Feature Maps
      │
      ▼
FPN + PAN Neck
      │
      ▼
Multi-scale Features
      │
      ▼
Detection Head
      │
      ▼
Bounding Boxes
Classes
Confidence Scores
      │
      ▼
Non-Maximum Suppression
      │
      ▼
Final Detections
```

---

# YOLOv8 vs Faster R-CNN

| Feature               | YOLOv8    | Faster R-CNN   |
| --------------------- | --------- | -------------- |
| Detector Type         | One-Stage | Two-Stage      |
| Speed                 | Very Fast | Slower         |
| Real-Time Capability  | Yes       | Limited        |
| Latency               | Low       | Higher         |
| Deployment Complexity | Low       | Higher         |
| Edge Deployment       | Easier    | More Difficult |

---

# Why YOLOv8 Was Selected For This Project

YOLOv8 was chosen because:

1. It provides real-time inference capability.
2. It achieves strong object detection performance.
3. It supports multi-scale detection required for BDD100K.
4. It is easy to train, evaluate, and deploy.
5. It is widely adopted in modern perception systems.

For autonomous driving datasets containing traffic lights, traffic signs, pedestrians, vehicles, and riders, YOLOv8 provides an effective balance between accuracy and computational efficiency.

---

# Key Takeaways

1. YOLOv8 is a one-stage detector.
2. Backbone extracts visual features.
3. Neck fuses multi-scale information using FPN and PAN.
4. Head predicts bounding boxes, classes, and confidence scores.
5. YOLOv8 uses anchor-free detection.
6. Multi-scale prediction enables detection of both small and large objects.
7. Non-Maximum Suppression removes duplicate detections.
8. YOLOv8 is well suited for real-time autonomous driving applications due to its low latency and strong accuracy-speed tradeoff.
