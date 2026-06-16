# Object Detection Metrics

## Objective

Understand the metrics used to evaluate object detection models and interpret YOLOv8 performance on the BDD100K dataset.

Evaluation metrics are critical because object detection requires both:

1. Correct classification
2. Accurate localization

A prediction is only considered correct if both conditions are satisfied.

---

# Evaluation Pipeline

```text
Ground Truth Boxes
        │
        ▼
Model Predictions
        │
        ▼
IoU Matching
        │
        ▼
True Positives
False Positives
False Negatives
        │
        ▼
Precision
Recall
AP
mAP
```

---

# Intersection over Union (IoU)

## Definition

IoU measures the overlap between the predicted bounding box and the ground-truth bounding box.

Formula:

```
IoU =
Intersection Area
-----------------
Union Area
```

Range:

| IoU | Meaning          |
| --- | ---------------- |
| 0.0 | No overlap       |
| 0.5 | Moderate overlap |
| 1.0 | Perfect overlap  |

---

# Example

Ground Truth:

```text
Car
```

Prediction:

```text
Car
```

Case 1:

Large overlap

```text
IoU = 0.85
```

Good detection.

Case 2:

Small overlap

```text
IoU = 0.20
```

Poor localization.

---

# IoU Threshold

A detection is considered correct only if:

```text
IoU ≥ Threshold
```

Common threshold:

```text
IoU ≥ 0.5
```

This is called:

```text
mAP@50
```

---

# True Positive (TP)

A prediction is a True Positive when:

1. Correct class
2. IoU exceeds threshold

Example:

```text
Ground Truth: Car
Prediction : Car
IoU        : 0.82
```

Result:

```text
True Positive
```

---

# False Positive (FP)

A False Positive occurs when:

* Object does not exist

or

* Wrong class is predicted

or

* IoU is too low

Example:

```text
Ground Truth: Car
Prediction : Truck
```

Result:

```text
False Positive
```

---

# False Negative (FN)

A False Negative occurs when the model misses an object.

Example:

```text
Ground Truth: Rider
Prediction : None
```

Result:

```text
False Negative
```

---

# Precision

## Definition

Precision measures prediction quality.

Formula:

```text
Precision =
TP
----------
TP + FP
```

Question answered:

```text
Of all predicted objects,
how many were correct?
```

---

# Example

Predictions:

```text
100 detections
```

Correct:

```text
90 detections
```

Incorrect:

```text
10 detections
```

Precision:

```text
90 / 100 = 0.90
```

Precision:

```text
90%
```

---

# High Precision Means

Few false alarms.

Example:

```text
Car detections are usually correct.
```

---

# Recall

## Definition

Recall measures detection coverage.

Formula:

```text
Recall =
TP
----------
TP + FN
```

Question answered:

```text
Of all actual objects,
how many were detected?
```

---

# Example

Ground Truth Cars:

```text
100
```

Detected:

```text
85
```

Missed:

```text
15
```

Recall:

```text
85 / 100 = 0.85
```

Recall:

```text
85%
```

---

# High Recall Means

Few missed objects.

Important for:

* Pedestrians
* Riders
* Traffic lights

Missing these objects can be safety-critical.

---

# Precision vs Recall

High Precision:

```text
Few false detections
```

High Recall:

```text
Few missed detections
```

Trade-off:

Increasing recall often decreases precision.

Increasing precision often decreases recall.

---

# Precision-Recall Curve

As confidence threshold changes:

```text
Precision
    │\
    │ \
    │  \
    │   \
    │    \
    └────────── Recall
```

The Precision-Recall curve summarizes detector performance across confidence thresholds.

---

# Average Precision (AP)

## Definition

Average Precision is the area under the Precision-Recall curve.

Range:

```text
0 → 1
```

or

```text
0% → 100%
```

---

# Per-Class AP

Each class has its own AP value.

Examples:

```text
AP(Car)
AP(Person)
AP(Traffic Light)
AP(Bus)
```

---

# Example

| Class | AP   |
| ----- | ---- |
| Car   | 0.92 |
| Truck | 0.85 |
| Rider | 0.61 |
| Train | 0.28 |

Interpretation:

Cars perform best.

Rare classes perform worse.

---

# Mean Average Precision (mAP)

## Definition

Mean Average Precision is the average AP across all classes.

Formula:

```text
mAP =
Average(AP of all classes)
```

---

# mAP@50

Uses:

```text
IoU = 0.50
```

A prediction is considered correct if:

```text
IoU ≥ 0.50
```

This is a relatively forgiving metric.

---

# mAP@50-95

Uses multiple thresholds:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

Average performance across all thresholds.

---

# Why mAP50-95 Is Harder

Example:

Prediction:

```text
IoU = 0.60
```

Counts as correct for:

```text
mAP50
```

May fail for:

```text
mAP75
mAP90
mAP95
```

Therefore:

```text
mAP50-95
```

is a stronger indicator of localization quality.

---

# Why mAP50-95 Matters

Two detectors may have identical:

```text
mAP50
```

but very different localization accuracy.

mAP50-95 reveals this difference.

---

# Expected Results for BDD100K

Based on dataset analysis:

Best-performing classes:

* Car
* Bus
* Truck

Reasons:

* Many training samples
* Large object sizes

Potentially weaker classes:

* Rider
* Motor
* Bike
* Train

Reasons:

* Class imbalance
* Heavy occlusion
* Small object sizes

---

# Confusion Matrix

A confusion matrix shows which classes are confused with one another.

Example:

| Actual | Predicted |
| ------ | --------- |
| Rider  | Bike      |
| Bike   | Motor     |
| Truck  | Bus       |

Useful for diagnosing model weaknesses.

---

# Interview Questions

## What is IoU?

IoU measures overlap between predicted and ground-truth bounding boxes.

---

## What is Precision?

Of all detections made by the model, how many were correct?

---

## What is Recall?

Of all real objects, how many were detected?

---

## What is AP?

Area under the Precision-Recall curve for a single class.

---

## What is mAP?

Average AP across all classes.

---

## Difference Between mAP50 and mAP50-95?

mAP50 evaluates detections at IoU=0.5 only.

mAP50-95 averages performance across IoU thresholds from 0.5 to 0.95 and is therefore more strict.

---

## Which Metric Is Most Important?

For object detection:

```text
mAP50-95
```

is generally considered the most comprehensive metric because it evaluates both classification accuracy and localization quality.

---

# Connection to BDD100K

Dataset analysis revealed:

* Severe class imbalance
* High occlusion
* Small-object dominance

These factors are expected to reduce:

* Recall for rare classes
* AP for heavily occluded objects
* mAP50-95 for small objects

Evaluation metrics provide quantitative evidence to validate these hypotheses.

---

# Key Takeaways

1. IoU measures localization accuracy.
2. Precision measures prediction quality.
3. Recall measures detection coverage.
4. AP is the area under the Precision-Recall curve.
5. mAP averages AP across classes.
6. mAP50 uses IoU=0.5.
7. mAP50-95 is stricter and more informative.
8. Confusion matrices help identify class-specific failure modes.
9. Metrics should always be interpreted together with dataset characteristics and failure analysis.
