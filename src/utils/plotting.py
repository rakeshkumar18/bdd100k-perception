import matplotlib.pyplot as plt


def plot_class_distribution(df, save_path):

    plt.figure(figsize=(12, 6))

    plt.bar(df["category"], df["count"])

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_bbox_area_histogram(df, save_path):

    plt.figure(figsize=(10, 6))

    plt.hist(df["area"], bins=50)

    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency")
    plt.title("BBox Area Distribution")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bbox_area_log_histogram(df, save_path):

    plt.figure(figsize=(10, 6))

    plt.hist(df["area"], bins=50, log=True)

    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency (log)")
    plt.title("BBox Area Distribution")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_aspect_ratio_histogram(df, save_path):

    plt.figure(figsize=(10, 6))

    plt.hist(df["aspect_ratio"], bins=50)

    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_occlusion_distribution(df, save_path):

    counts = df["occluded"].value_counts()

    plt.figure(figsize=(6, 6))

    plt.pie(counts, labels=["Visible", "Occluded"], autopct="%1.1f%%")

    plt.title("Occlusion Distribution")

    plt.savefig(save_path)

    plt.close()


def plot_occlusion_by_class(occlusion_df, save_path):

    plt.figure(figsize=(12, 6))

    plt.bar(occlusion_df["category"], occlusion_df["occluded"])

    plt.xticks(rotation=45)

    plt.ylabel("Occlusion Percentage")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_category_distribution(df, category_col, count_col, title, save_path):

    plt.figure(figsize=(12, 6))

    plt.bar(df[category_col], df[count_col])

    plt.xticks(rotation=45)

    plt.title(title)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_train_val_comparison(df, category_col, save_path, title):

    import matplotlib.pyplot as plt
    import numpy as np

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
