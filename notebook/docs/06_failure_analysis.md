# Failure Analysis and Model Improvement Strategy

## Objective

Analyze YOLOv8 validation results on the BDD100K dataset, identify key failure modes, validate hypotheses from exploratory data analysis (EDA), and propose improvements for future model iterations.

---

# Evaluation Summary

Validation was performed on the BDD100K validation dataset containing 10,000 images and 185,526 annotated objects.

## Overall Performance

| Metric    | Value |
| --------- | ----: |
| Precision | 0.437 |
| Recall    | 0.297 |
| mAP50     | 0.295 |
| mAP50-95  | 0.155 |

---

# Initial Observations

The model achieves higher precision than recall.

| Metric    | Value |
| --------- | ----: |
| Precision | 43.7% |
| Recall    | 29.7% |

This indicates that the detector is relatively conservative.

### Interpretation

The model produces fewer incorrect detections but misses a significant number of objects.

```text
Recall < Precision
```

Therefore:

* False negatives are a larger issue than false positives.
* Detection coverage is limited.
* Small and occluded objects are frequently missed.

---

# EDA Predictions vs Evaluation Results

Prior to training, exploratory data analysis identified several expected challenges.

| EDA Prediction                    | Validation Result |
| --------------------------------- | ----------------- |
| Cars should perform best          | ✅ Confirmed       |
| Trains should perform worst       | ✅ Confirmed       |
| Riders should be difficult        | ✅ Confirmed       |
| Motorcycles should be difficult   | ✅ Confirmed       |
| Small objects will be challenging | ✅ Confirmed       |
| Occluded classes will struggle    | ✅ Confirmed       |

The evaluation results strongly support the hypotheses generated during dataset analysis.

---

# Class-Wise Performance Analysis

## Car

| Metric    | Value |
| --------- | ----: |
| Precision | 0.603 |
| Recall    | 0.628 |
| mAP50     | 0.639 |
| mAP50-95  | 0.374 |

### Analysis

Cars are the best-performing class.

Reasons:

* Largest class in the dataset
* Large object size
* High visual consistency
* Lower sensitivity to localization errors

### Conclusion

The detector learns robust car representations due to the abundance of training samples.

---

## Truck

| Metric    | Value |
| --------- | ----: |
| Precision | 0.420 |
| Recall    | 0.437 |
| mAP50     | 0.384 |
| mAP50-95  | 0.254 |

### Analysis

Truck performance is noticeably lower than cars.

Reasons:

* Significantly fewer training examples
* Greater variation in appearance
* Lower representation across scenes

---

## Bus

| Metric    | Value |
| --------- | ----: |
| Precision | 0.379 |
| Recall    | 0.375 |
| mAP50     | 0.345 |
| mAP50-95  | 0.258 |

### Analysis

Bus detection is moderate.

Although buses are large objects, the class contains relatively few examples compared to cars.

---

## Person

| Metric    | Value |
| --------- | ----: |
| Precision | 0.460 |
| Recall    | 0.425 |
| mAP50     | 0.400 |
| mAP50-95  | 0.178 |

### Analysis

Person detection is significantly harder than vehicle detection.

Reasons:

* Smaller object sizes
* Frequent partial occlusions
* Large pose variation

---

## Rider

| Metric    | Value |
| --------- | ----: |
| Precision | 0.679 |
| Recall    | 0.111 |
| mAP50     | 0.193 |
| mAP50-95  | 0.082 |

### Analysis

Rider is one of the most challenging classes.

The model detects riders accurately when detected, but misses the majority of rider instances.

### Root Cause

EDA revealed:

* Rider occlusion rate: 89.16%
* Training samples: 4,521

This combination severely impacts recall.

---

## Bike

| Metric    | Value |
| --------- | ----: |
| Precision | 0.430 |
| Recall    | 0.157 |
| mAP50     | 0.164 |
| mAP50-95  | 0.070 |

### Analysis

Bike performance is poor.

Likely causes:

* Small object size
* Heavy occlusion
* Limited number of training examples

---

## Motorcycle

| Metric    | Value |
| --------- | ----: |
| Precision | 0.446 |
| Recall    | 0.036 |
| mAP50     | 0.084 |
| mAP50-95  | 0.039 |

### Analysis

Motorcycle is among the worst-performing classes.

### Root Cause

EDA showed:

* 3,002 training samples
* 76.5% occlusion rate

The model struggles to learn a stable representation for this class.

---

## Traffic Light

| Metric    | Value |
| --------- | ----: |
| Precision | 0.462 |
| Recall    | 0.388 |
| mAP50     | 0.340 |
| mAP50-95  | 0.106 |

### Analysis

Traffic lights are detected reasonably often but localized poorly.

Observation:

```text
mAP50 = 0.340
mAP50-95 = 0.106
```

The large drop indicates localization errors.

### Root Cause

Traffic lights occupy very few pixels and are often distant from the camera.

---

## Traffic Sign

| Metric    | Value |
| --------- | ----: |
| Precision | 0.486 |
| Recall    | 0.415 |
| mAP50     | 0.397 |
| mAP50-95  | 0.189 |

### Analysis

Traffic signs outperform traffic lights.

Likely reason:

* Larger average object size
* Better visual structure
* More training examples

---

## Train

| Metric    | Value |
| --------- | ----: |
| Precision | 0.000 |
| Recall    | 0.000 |
| mAP50     | 0.000 |
| mAP50-95  | 0.000 |

### Analysis

The model completely fails to detect trains.

### Root Cause

EDA identified:

* Only 136 train instances in the training set
* Only 15 train instances in validation

The class is too underrepresented for the model to learn meaningful features.

---

# Impact of Dataset Imbalance

Dataset statistics revealed:

| Class |   Count |
| ----- | ------: |
| Car   | 713,917 |
| Train |     136 |

Imbalance ratio:

```text
5249 : 1
```

The evaluation confirms that dataset imbalance directly impacts detection performance.

---

# Small Object Failure Analysis

The following classes contain a large proportion of small objects:

* Traffic Light
* Traffic Sign
* Rider
* Bike
* Motorcycle

Observed results:

* Lower recall
* Lower mAP50-95
* Higher localization error

### Root Cause

Small objects occupy fewer pixels and produce weaker feature representations in deeper network layers.

---

# Occlusion Failure Analysis

EDA Results:

| Metric           |   Value |
| ---------------- | ------: |
| Occluded Objects | 609,273 |
| Occlusion Rate   |   47.3% |

Most occluded classes:

* Rider
* Bike
* Motorcycle

These are also the worst-performing classes during evaluation.

### Conclusion

There is a strong correlation between occlusion rate and detection performance.

---

# Why Overall mAP Is Low

The primary reasons are:

## Class Imbalance

Extremely uneven class distribution reduces performance on minority classes.

## Heavy Occlusion

Nearly half of all objects are partially occluded.

## Small Objects

Traffic lights and traffic signs are difficult to localize accurately.

## Rare Classes

Train, motorcycle, rider, and bike classes lack sufficient training data.

---

# Training Constraints

The model was trained using:

```yaml
epochs: 1
model: YOLOv8n
```

This represents an intentionally short experimental run.

Therefore, the current metrics should be interpreted as a baseline rather than final performance.

---

# Improvement Strategy

## Data-Centric Improvements

### Increase Minority-Class Representation

Focus on:

* Train
* Motorcycle
* Rider
* Bike

### Class-Aware Sampling

Use weighted or balanced sampling to reduce class imbalance.

### Additional Data Collection

Acquire more examples for underrepresented classes.

---

## Model Improvements

### Train Longer

Recommended:

```yaml
epochs: 50
```

or

```yaml
epochs: 100
```

### Larger Architecture

Evaluate:

* YOLOv8s
* YOLOv8m

to improve feature extraction capacity.

---

## Augmentation Improvements

Consider enabling:

* Mosaic augmentation
* MixUp augmentation
* Copy-Paste augmentation

These techniques improve robustness for small and rare objects.

---

## Inference Improvements

Experiment with:

* Confidence threshold tuning
* NMS threshold tuning
* Test-Time Augmentation (TTA)

---

# Key Lessons Learned

1. Dataset characteristics strongly influence detector performance.
2. EDA successfully predicted the major failure modes before training.
3. Class imbalance remains the largest challenge in BDD100K.
4. Occlusion significantly reduces recall.
5. Small-object detection remains difficult for lightweight YOLO models.
6. Rare classes require additional data or sampling strategies.
7. Evaluation should always be paired with root-cause analysis rather than reporting metrics alone.

---

# Conclusion

The validation results confirm the challenges identified during exploratory data analysis. Classes with severe imbalance, high occlusion rates, and small object sizes demonstrate substantially lower recall and mAP. Cars, which dominate the dataset, achieve the strongest performance.

This analysis provides a clear roadmap for future improvements through better class balancing, longer training schedules, stronger augmentation strategies, and larger model architectures.
