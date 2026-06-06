from pathlib import Path
import pandas as pd

from src.ingestion.parser import BDDParser
from src.ingestion.dataframe_builder import DataFrameBuilder

from src.analysis.class_stats import ClassStats
from src.analysis.bbox_stats import BBoxStats
from src.analysis.occlusion_stats import OcclusionStats
from src.analysis.scene_stats import SceneStats
from src.analysis.train_val_comparison import TrainValComparison

from src.utils.paths import (
    TRAIN_LABELS,
    VAL_LABELS,
    REPORT_DIR,
)

from src.utils.plotting import (
    plot_class_distribution,
    plot_bbox_area_histogram,
    plot_bbox_area_log_histogram,
    plot_aspect_ratio_histogram,
    plot_occlusion_distribution,
    plot_occlusion_by_class,
    plot_category_distribution,
    plot_train_val_comparison,
)

FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def build_split_dataframe(label_path, output_filename):

    print(f"\nProcessing {output_filename}")

    parser = BDDParser()

    scenes = parser.load_directory(label_path)

    print(f"Loaded {len(scenes):,} scenes")

    builder = DataFrameBuilder()

    df = builder.build(scenes)

    print(f"Generated {len(df):,} objects")

    output_file = REPORT_DIR / output_filename

    builder.save_csv(df, output_file)

    print(f"Saved: {output_file}")

    return df


def analyze_split(df, split_name="train"):

    print(f"\nAnalyzing {split_name}")

    report_dir = REPORT_DIR
    figure_dir = FIGURE_DIR

    class_stats = ClassStats(df)

    class_dist = class_stats.class_distribution()

    class_dist.to_csv(report_dir / f"{split_name}_class_distribution.csv", index=False)

    plot_class_distribution(
        class_dist, figure_dir / f"{split_name}_class_distribution.png"
    )

    print(class_stats.summary())

    bbox_stats = BBoxStats(df)

    print(bbox_stats.area_summary())

    print(bbox_stats.size_distribution())

    plot_bbox_area_histogram(df, figure_dir / f"{split_name}_bbox_area_histogram.png")

    plot_bbox_area_log_histogram(
        df, figure_dir / f"{split_name}_bbox_area_log_histogram.png"
    )

    plot_aspect_ratio_histogram(
        df, figure_dir / f"{split_name}_aspect_ratio_histogram.png"
    )

    occ_stats = OcclusionStats(df)

    print(occ_stats.overall_occlusion_rate())

    print(occ_stats.truncation_rate())

    occ_df = occ_stats.occlusion_by_class()

    plot_occlusion_distribution(
        df, figure_dir / f"{split_name}_occlusion_distribution.png"
    )

    plot_occlusion_by_class(occ_df, figure_dir / f"{split_name}_occlusion_by_class.png")

    scene_stats = SceneStats(df)

    weather_df = scene_stats.weather_distribution()

    time_df = scene_stats.timeofday_distribution()

    scene_df = scene_stats.scene_distribution()

    weather_df.to_csv(
        report_dir / f"{split_name}_weather_distribution.csv", index=False
    )

    time_df.to_csv(report_dir / f"{split_name}_timeofday_distribution.csv", index=False)

    scene_df.to_csv(report_dir / f"{split_name}_scene_distribution.csv", index=False)

    plot_category_distribution(
        weather_df,
        "weather",
        "count",
        f"{split_name} Weather Distribution",
        figure_dir / f"{split_name}_weather_distribution.png",
    )

    plot_category_distribution(
        time_df,
        "timeofday",
        "count",
        f"{split_name} Time Distribution",
        figure_dir / f"{split_name}_timeofday_distribution.png",
    )

    plot_category_distribution(
        scene_df,
        "scene",
        "count",
        f"{split_name} Scene Distribution",
        figure_dir / f"{split_name}_scene_distribution.png",
    )


def compare_train_val(train_df, val_df):

    comparison = TrainValComparison(train_df, val_df)

    # ----------------------------------
    # Class Comparison
    # ----------------------------------

    class_cmp = comparison.compare_classes()

    class_cmp.to_csv(REPORT_DIR / "train_val_class.csv", index=False)

    plot_train_val_comparison(
        class_cmp,
        "category",
        FIGURE_DIR / "train_val_class.png",
        "Train vs Validation Classes",
    )

    # ----------------------------------
    # Weather Comparison
    # ----------------------------------

    weather_cmp = comparison.compare_column("weather")

    weather_cmp.to_csv(REPORT_DIR / "train_val_weather.csv", index=False)

    plot_train_val_comparison(
        weather_cmp,
        "weather",
        FIGURE_DIR / "train_val_weather.png",
        "Train vs Validation Weather",
    )

    # ----------------------------------
    # Scene Comparison
    # ----------------------------------

    scene_cmp = comparison.compare_column("scene")

    scene_cmp.to_csv(REPORT_DIR / "train_val_scene.csv", index=False)

    plot_train_val_comparison(
        scene_cmp,
        "scene",
        FIGURE_DIR / "train_val_scene.png",
        "Train vs Validation Scene",
    )

    # ----------------------------------
    # Time Of Day Comparison
    # ----------------------------------

    timeofday_cmp = comparison.compare_column("timeofday")

    timeofday_cmp.to_csv(REPORT_DIR / "train_val_timeofday.csv", index=False)

    plot_train_val_comparison(
        timeofday_cmp,
        "timeofday",
        FIGURE_DIR / "train_val_timeofday.png",
        "Train vs Validation Time Of Day",
    )

    print("\nTrain-Val Comparison Complete")


def data_quality_report(df):

    print("\nDATA QUALITY REPORT")
    print("=" * 50)

    print("Zero area:", (df["area"] == 0).sum())

    print("Negative area:", (df["area"] < 0).sum())

    print("Zero width:", (df["width"] == 0).sum())

    print("Zero height:", (df["height"] == 0).sum())

    print("Negative width:", (df["width"] < 0).sum())

    print("Negative height:", (df["height"] < 0).sum())


def main():

    train_df = build_split_dataframe(TRAIN_LABELS, "train_objects.csv")

    val_df = build_split_dataframe(VAL_LABELS, "val_objects.csv")

    print("\nValidation Checks")

    print(train_df.info())

    print(train_df["category"].value_counts().head())

    print(train_df.isnull().sum())

    analyze_split(train_df, "train")

    analyze_split(val_df, "val")

    compare_train_val(train_df, val_df)

    data_quality_report(train_df)
    data_quality_report(val_df)


if __name__ == "__main__":
    main()
