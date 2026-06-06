# BDD100K Perception Pipeline using YOLOv8

## Overview

This project implements an end-to-end computer vision perception pipeline using the BDD100K autonomous driving dataset and YOLOv8 object detection models.

The goal is to build a production-style machine learning workflow that goes beyond model training and includes:

* Dataset analysis
* Data visualization
* Object detection training
* Experiment tracking
* Model evaluation
* Inference services
* Interactive dashboards
* Docker deployment

## Assignment Scope

This repository includes a dedicated analysis of the BDD100K dataset for the **object detection** task only.

Included:
- 10 object-detection classes with `box2d` annotations:
  - bike
  - bus
  - car
  - motor
  - person
  - rider
  - traffic light
  - traffic sign
  - train
  - truck
- Train split
- Validation split

Excluded:
- Test split
- Drivable area annotations
- Lane marking annotations
- Semantic segmentation labels
- Any non-detection tasks

---

## Features

### Dataset Analysis

* BDD100K annotation parsing
* Class distribution analysis
* Bounding box statistics
* Weather distribution analysis
* Scene distribution analysis
* Time-of-day analysis
* Train vs Validation comparison

Generated visualizations include:

* Class distribution
* Bounding box area histograms
* Aspect ratio distributions
* Weather distributions
* Scene distributions
* Time-of-day distributions

---

### YOLOv8 Training

Supported model variants:

* YOLOv8n
* YOLOv8s
* YOLOv8m
* YOLOv8l
* YOLOv8x

Training features:

* Configurable epochs
* Configurable image size
* Configurable batch size
* Apple Silicon MPS support
* Automatic artifact generation

Training outputs:

* Best model weights
* Last model weights
* Precision-Recall curves
* F1 curves
* Confusion matrices
* Training history

---

### MLflow Integration

Experiment tracking includes:

* Hyperparameters
* Training configuration
* Metrics
* Artifacts

Tracked metrics:

* Precision
* Recall
* mAP50
* mAP50-95

---

### Streamlit Dashboard

Interactive dashboard pages:

#### Training Dashboard

* Training metrics
* Run comparison
* Training artifacts
* Model performance visualization

#### Dataset Analysis

* Dataset EDA visualizations
* Train/Validation comparisons

#### Inference

* Upload image
* Run object detection
* View predictions
* Detection statistics

#### Evaluation

* Model performance metrics
* Precision-recall curves
* Confusion matrices

---

### Inference Pipeline

Single image prediction

Features:

* Object detection
* Bounding box rendering
* Confidence scores
* Detection summaries

---

### Batch Prediction

Batch inference support for directories of images.

Outputs:

* Annotated images
* Detection results
* Prediction statistics

---

### Docker Support

Containerized deployment using:

* Docker
* Docker Compose

---

## Project Structure

```text
.
├── Dockerfile
├── Project_Readme.md
├── README.md
├── configs
│   ├── dataset.yaml
│   ├── training.yaml
│   └── yolo_dataset.yaml
├── docker-compose.yml
├── mlflow.db
├── outputs
│   ├── figures
│   ├── mlflow
│   ├── predictions
│   └── reports
├── pyproject.toml
├── pytest.ini
├── requirements-dev.txt
├── requirements.txt
├── runs
│   └── detect
├── scripts
│   ├── __init__.py
│   ├── analyze_dataset.py
│   ├── batch_predict.py
│   ├── check_runs.py
│   ├── convert_to_yolo.py
│   ├── evaluate.py
│   ├── get_best_model.py
│   ├── inspect_label.py
│   ├── predict.py
│   ├── test_paths.py
│   ├── train_yolo.py
│   └── validate_dataset.py
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── analysis
│   ├── dashboard
│   ├── dataset
│   ├── evaluation
│   ├── inference
│   ├── ingestion
│   ├── model_registry
│   ├── tracking
│   ├── training
│   └── utils
├── tests
│   ├── test.py
│   ├── test_dataframe_builder.py
│   ├── test_parser.py
│   ├── test_path.py
│   └── test_schema.py
└── yolov8n.pt
```

---

## Installation

### Create Environment

```bash
python -m venv assignment
source assignment/bin/activate
```
### Install project in editable mode
```bash
python -m pip install -e .
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

---
## EDA

```bash
python -m scripts.analyze_dataset
```
### This will genearte the figure and csv for EDA analysis.
---
## Training (for training keep the train and val data at the root of bdd100k-perception folder)

```bash
python -m scripts.train_yolo
```

---


## Evaluation

```bash
python -m scripts.evaluate_model
```

---

## Batch Prediction

```bash
python -m scripts.batch_predict
```

---

## Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Docker

Build:

```bash
docker compose build
```

Run:

```bash
docker compose up
```

Dashboard:

```text
http://localhost:8501
```
## Update docker after change

```bash
docker compose down
docker compose build --no-cache
docker compose up
```
---

## Dataset

Dataset used:

BDD100K

Categories:

* Car
* Truck
* Bus
* Person
* Rider
* Bike
* Motor
* Traffic Light
* Traffic Sign
* Train

---

## Future Improvements

### Model Improvements

* YOLOv11 migration
* Hyperparameter optimization
* Cross-validation
* Data augmentation experiments
* Multi-model comparison

### MLOps

* MLflow Model Registry
* Automated best-model selection
* CI/CD pipelines
* DVC integration
* Automated retraining

### Deployment

* REST API
* FastAPI serving
* Kubernetes deployment
* Cloud deployment

### Advanced Perception

* Lane detection
* Semantic segmentation
* Multi-object tracking
* Sensor fusion
* Autonomous driving stack integration

---

## Author

Rakesh Kumar

Computer Vision | Machine Learning | Autonomous Systems
