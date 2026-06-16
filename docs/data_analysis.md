# BDD100K Dataset Analysis

## Objective

Analyze the BDD100K object detection dataset to understand class distribution, scene characteristics, annotation quality, and potential challenges affecting model performance.

---

# Dataset Processing Pipeline

The dataset ingestion pipeline was designed as:

BDD100K JSON Annotations
→ Parser
→ Dataclasses (SceneAnnotation, ObjectAnnotation, BoundingBox)
→ DataFrame Builder
→ EDA Reports
→ Streamlit Dashboard

The DataFrameBuilder converts scene-level annotations into an object-level tabular dataset where each row represents a single object instance.

Generated features:

* Bounding box width
* Bounding box height
* Bounding box area
* Aspect ratio
* Occlusion flag
* Truncation flag

Invalid bounding boxes are filtered and stored separately for quality analysis.

---

# Dataset Statistics

## Total Objects

Total annotated objects:

1,288,010

## Number of Classes

10 object classes.

---

# Class Distribution

| Class         |   Count |
| ------------- | ------: |
| Car           | 713,917 |
| Traffic Sign  | 239,893 |
| Traffic Light | 186,224 |
| Person        |  91,405 |
| Truck         |  30,003 |
| Bus           |  11,684 |
| Bike          |   7,225 |
| Rider         |   4,521 |
| Motor         |   3,002 |
| Train         |     136 |

## Key Observation

The dataset exhibits severe class imbalance.

Largest class:

* Car: 713,917

Smallest class:

* Train: 136

Imbalance ratio:

5249:1

### Expected Impact

Classes with very few samples are expected to have lower recall and poorer generalization.

Potentially affected classes:

* Train
* Motor
* Rider
* Bike

---

# Weather Distribution

| Weather       |   Count |
| ------------- | ------: |
| Clear         | 653,843 |
| Overcast      | 185,140 |
| Undefined     | 161,023 |
| Snowy         |  99,680 |
| Partly Cloudy |  97,018 |
| Rainy         |  89,446 |
| Foggy         |   1,860 |

## Key Observation

Clear weather dominates the dataset.

Foggy scenes are extremely underrepresented.

### Expected Impact

The detector is expected to perform best under clear conditions and may experience degraded performance in foggy environments due to limited training examples.

---

# Scene Distribution

| Scene       |   Count |
| ----------- | ------: |
| City Street | 890,290 |
| Highway     | 253,983 |
| Residential | 133,448 |
| Parking Lot |   5,655 |
| Undefined   |   3,471 |
| Tunnel      |     880 |
| Gas Station |     283 |

## Key Observation

City-street driving dominates the dataset.

Tunnel and gas-station scenes are extremely rare.

### Expected Impact

The detector may generalize poorly to underrepresented scene types.

---

# Occlusion Analysis

Total Objects:

1,288,010

Occluded Objects:

609,273

Occlusion Rate:

47.3%

## Most Occluded Classes

| Class | Occlusion Rate (%) |
| ----- | -----------------: |
| Rider |              89.16 |
| Bike  |              83.82 |
| Motor |              76.52 |
| Car   |              67.73 |
| Bus   |              65.52 |

## Key Observation

Nearly half of all objects are partially occluded.

### Expected Impact

Occlusion is expected to increase:

* False negatives
* Localization errors
* Detection difficulty for riders and motorcycles

---

# Truncation Analysis

Truncated Objects:

89,341

Truncation Rate:

6.94%

## Key Observation

Occlusion is a much larger challenge than truncation.

Most detection difficulty originates from object overlap rather than objects leaving image boundaries.

---

# Object Size Analysis

Area Statistics:

* Mean Area: 6,785
* Median Area: 818
* Maximum Area: 917,709

## Key Observation

Object-size distribution is heavily skewed.

Most objects are small while a few very large objects increase the mean significantly.

### Expected Impact

Small objects are expected to be more difficult to detect.

Potentially affected categories:

* Traffic lights
* Traffic signs
* Distant pedestrians
* Riders
* Motorcycles

---

# Dataset Bias

The dataset is biased toward:

* Cars
* Clear weather
* City-street environments

Underrepresented conditions:

* Fog
* Trains
* Gas stations
* Tunnels
* Motorcycles
* Riders

This bias is expected to influence detector performance.

---

# Predicted Failure Modes

Based on EDA, the following challenges are expected before model training:

1. Poor recall on train class due to extremely low sample count.
2. Reduced rider and motorcycle detection performance due to severe occlusion.
3. Difficulty detecting small traffic lights and traffic signs.
4. Performance degradation in foggy conditions.
5. Reduced robustness in tunnel and gas-station scenes.

---

# Conclusion

The BDD100K dataset provides a realistic driving dataset with significant class imbalance, heavy occlusion, environmental bias, and a large number of small objects.

These findings establish clear hypotheses for model evaluation and failure analysis in subsequent stages of the project.
