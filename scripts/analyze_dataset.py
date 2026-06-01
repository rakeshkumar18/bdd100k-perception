from src.ingestion.parser import BDDParser
from src.ingestion.dataframe_builder import (
    DataFrameBuilder
)

from src.utils.paths import (
    TRAIN_LABELS,
    REPORT_DIR,
)

parser = BDDParser()

print("Loading labels...")

scenes = parser.load_directory(
    TRAIN_LABELS,
    max_files=100
)

print(
    f"Loaded {len(scenes)} scenes"
)

builder = DataFrameBuilder()

df = builder.build(scenes)

print(df.head())

output_file = (
    REPORT_DIR /
    "train_objects_sample.csv"
)

builder.save_csv(
    df,
    output_file
)

print(
    f"Saved: {output_file}"
)

from pathlib import Path

sample = df.iloc[0]

print(sample["image_name"])

label_file = (
    TRAIN_LABELS /
    f"{sample['image_name']}.json"
)

print(label_file)
print(label_file.exists())