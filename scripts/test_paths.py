# scripts/test_paths.py

from src.utils.paths import *

print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATASET_ROOT =", DATASET_ROOT)

print("TRAIN_IMAGES =", TRAIN_IMAGES.exists())
print("VAL_IMAGES =", VAL_IMAGES.exists())

print("TRAIN_LABELS =", TRAIN_LABELS.exists())
print("VAL_LABELS =", VAL_LABELS.exists())
