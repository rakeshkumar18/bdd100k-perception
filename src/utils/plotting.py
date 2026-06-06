"""Plotting utilities for BDD100K object detection dataset analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_class_distribution(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot class distribution as a bar chart and save to file.

    Args:
        df: DataFrame with 'category' and 'count' columns.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df["category"], df["count"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bbox_area_histogram(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot bounding box area histogram and save to file.

    Args:
        df: DataFrame with 'area' column.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["area"], bins=50)
    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency")
    plt.title("BBox Area Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bbox_area_log_histogram(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot bounding box area histogram with log scale and save to file.

    Args:
        df: DataFrame with 'area' column.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["area"], bins=50, log=True)
    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency (log)")
    plt.title("BBox Area Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_aspect_ratio_histogram(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot aspect ratio histogram and save to file.

    Args:
        df: DataFrame with 'aspect_ratio' column.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["aspect_ratio"], bins=50)
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_occlusion_distribution(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot occlusion distribution as a pie chart and save to file.

    Args:
        df: DataFrame with 'occluded' column (boolean or 0/1).
        save_path: Output file path for the plot image.
    """
    counts = df["occluded"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=["Visible", "Occluded"], autopct="%1.1f%%")
    plt.title("Occlusion Distribution")
    plt.savefig(save_path)
    plt.close()


def plot_occlusion_by_class(occlusion_df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot occlusion percentage by class as a bar chart and save to file.

    Args:
        occlusion_df: DataFrame with 'category' and 'occluded' columns.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(occlusion_df["category"], occlusion_df["occluded"])
    plt.xticks(rotation=45)
    plt.ylabel("Occlusion Percentage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_category_distribution(
    df: pd.DataFrame,
    category_col: str,
    count_col: str,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot category distribution as a bar chart and save to file.

    Args:
        df: DataFrame with category and count columns.
        category_col: Name of the category column.
        count_col: Name of the count column.
        title: Plot title.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df[category_col], df[count_col])
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_train_val_comparison(
    df: pd.DataFrame, category_col: str, save_path: str | Path, title: str
) -> None:
    """Plot train vs validation percentage comparison and save to file.

    Args:
        df: DataFrame with 'train_pct', 'val_pct', and category column.
        category_col: Name of the category column.
        save_path: Output file path for the plot image.
        title: Plot title.
    """
    x = np.arange(len(df))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, df["train_pct"], width, label="Train")
    plt.bar(x + width / 2, df["val_pct"], width, label="Validation")
    plt.xticks(x, df[category_col], rotation=45)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    """Plotting utilities for BDD100K object detection dataset analysis."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_class_distribution(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot class distribution as a bar chart and save to file.

    Args:
        df: DataFrame with 'category' and 'count' columns.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df["category"], df["count"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bbox_area_histogram(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot bounding box area histogram and save to file.

    Args:
        df: DataFrame with 'area' column.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["area"], bins=50)
    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency")
    plt.title("BBox Area Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bbox_area_log_histogram(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot bounding box area histogram with log scale and save to file.

    Args:
        df: DataFrame with 'area' column.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["area"], bins=50, log=True)
    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency (log)")
    plt.title("BBox Area Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_aspect_ratio_histogram(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot aspect ratio histogram and save to file.

    Args:
        df: DataFrame with 'aspect_ratio' column.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df["aspect_ratio"], bins=50)
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_occlusion_distribution(df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot occlusion distribution as a pie chart and save to file.

    Args:
        df: DataFrame with 'occluded' column (boolean or 0/1).
        save_path: Output file path for the plot image.
    """
    counts = df["occluded"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=["Visible", "Occluded"], autopct="%1.1f%%")
    plt.title("Occlusion Distribution")
    plt.savefig(save_path)
    plt.close()


def plot_occlusion_by_class(occlusion_df: pd.DataFrame, save_path: str | Path) -> None:
    """Plot occlusion percentage by class as a bar chart and save to file.

    Args:
        occlusion_df: DataFrame with 'category' and 'occluded' columns.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(occlusion_df["category"], occlusion_df["occluded"])
    plt.xticks(rotation=45)
    plt.ylabel("Occlusion Percentage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_category_distribution(
    df: pd.DataFrame,
    category_col: str,
    count_col: str,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot category distribution as a bar chart and save to file.

    Args:
        df: DataFrame with category and count columns.
        category_col: Name of the category column.
        count_col: Name of the count column.
        title: Plot title.
        save_path: Output file path for the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df[category_col], df[count_col])
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_train_val_comparison(
    df: pd.DataFrame, category_col: str, save_path: str | Path, title: str
) -> None:
    """Plot train vs validation percentage comparison and save to file.

    Args:
        df: DataFrame with 'train_pct', 'val_pct', and category column.
        category_col: Name of the category column.
        save_path: Output file path for the plot image.
        title: Plot title.
    """
    x = np.arange(len(df))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, df["train_pct"], width, label="Train")
    plt.bar(x + width / 2, df["val_pct"], width, label="Validation")
    plt.xticks(x, df[category_col], rotation=45)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
