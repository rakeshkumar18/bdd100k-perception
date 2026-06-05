# BDD100K Perception Pipeline

## Technical Project Report

---

# 1. Introduction

Autonomous driving systems require robust perception pipelines capable of detecting and understanding objects within a driving scene.

This project focuses on building a scalable object detection pipeline using the BDD100K dataset and YOLOv8 architecture while incorporating experiment tracking, visualization, evaluation, and deployment capabilities.

The objective was not only to train a detector but also to build an engineering workflow resembling a production machine learning system.

---

# 2. Project Objectives

The primary objectives were:

1. Understand BDD100K dataset characteristics.
2. Build automated dataset analysis tools.
3. Train object detection models.
4. Track experiments systematically.
5. Build evaluation workflows.
6. Develop an interactive dashboard.
7. Support inference workflows.
8. Containerize the application.

---

# 3. Dataset Analysis

Dataset:

BDD100K

The following analyses were performed:

### Class Distribution

Investigated class imbalance among:

* Cars
* Trucks
* Buses
* Riders
* Bikes
* Motorcycles
* Traffic signs
* Traffic lights

Observations:

* Cars dominate the dataset.
* Trains are extremely rare.
* Significant class imbalance exists.

---

### Bounding Box Statistics

Computed:

* Area distribution
* Log-area distribution
* Aspect ratio distribution

Findings:

* Majority of objects occupy small image regions.
* Long-tail distribution observed.
* Detection of small objects remains challenging.

---

### Environmental Attributes

Analyzed:

* Weather
* Scene
* Time of day

Findings:

* Most samples correspond to daytime conditions.
* Certain weather conditions are underrepresented.
* Scene diversity supports robust training.

---

# 4. Model Development

Model:

YOLOv8n

Reasons for selection:

* Fast training
* Lightweight architecture
* Suitable for experimentation
* Good baseline performance

Training configuration:

* Image size: 640
* Batch size: 8
* Device: Apple Silicon MPS

---

# 5. Experiment Tracking

MLflow was integrated to track:

### Parameters

* Model
* Epochs
* Batch size
* Image size
* Device

### Metrics

* Precision
* Recall
* mAP50
* mAP50-95

### Artifacts

* Training curves
* Confusion matrices
* Model weights

Benefits:

* Reproducibility
* Run comparison
* Experiment management

---

# 6. Evaluation Results

Best validation results obtained:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.636 |
| Recall    | 0.363 |
| mAP50     | 0.393 |
| mAP50-95  | 0.212 |

Observations:

* Strong performance on cars.
* Lower performance on rare classes.
* Small object detection remains challenging.

---

# 7. Dashboard Development

Streamlit dashboard provides:

### Training Dashboard

* KPI monitoring
* Run comparison
* Training curves
* Artifact visualization

### Dataset Analysis

* Dataset visualizations
* Statistical summaries

### Inference

* Interactive image uploads
* Detection visualization
* Detection tables

### Evaluation

* Precision-recall analysis
* Confusion matrix review

---

# 8. Inference Pipeline

Implemented:

### Single Image Inference

Workflow:

1. Load trained model.
2. Predict objects.
3. Render detections.
4. Generate summaries.

### Batch Inference

Workflow:

1. Load directory.
2. Run predictions.
3. Save annotated images.
4. Generate outputs.

---

# 9. Dockerization

Docker deployment was added to support:

* Environment consistency
* Reproducibility
* Simplified deployment

Challenges encountered:

* OpenCV dependencies
* libGL requirements
* Container image optimization

---

# 10. Software Engineering Practices

Applied practices:

* Modular architecture
* Separation of concerns
* Configuration-driven execution
* Experiment tracking
* Dashboard monitoring
* Reusable components

---

# 11. Lessons Learned

Key lessons:

1. Dataset quality directly impacts detector performance.
2. Experiment tracking becomes essential after multiple runs.
3. Visualization accelerates debugging.
4. Dockerization exposes hidden dependency issues.
5. Production workflows require more than model training.

---

# 12. Future Roadmap

Short-Term:

* Model Registry
* Automatic Best Model Selection
* Improved dashboard analytics
* Multi-run comparisons

Medium-Term:

* Hyperparameter optimization
* DVC integration
* Automated evaluation

Long-Term:

* Tracking
* Segmentation
* Lane detection
* Sensor fusion
* Full autonomous perception stack

---

# Conclusion

This project successfully evolved from a YOLO training exercise into a complete perception workflow incorporating dataset analysis, model training, experiment tracking, evaluation, visualization, inference, and deployment.

The resulting framework provides a strong foundation for future research, experimentation, and production-oriented computer vision development.
