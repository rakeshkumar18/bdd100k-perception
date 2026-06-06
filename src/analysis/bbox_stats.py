import pandas as pd


class BBoxStats:

    def __init__(self, df):
        self.df = df

    def area_summary(self):

        return {
            "mean_area": self.df["area"].mean(),
            "median_area": self.df["area"].median(),
            "min_area": self.df["area"].min(),
            "max_area": self.df["area"].max(),
        }

    def size_distribution(self):

        bins = [0, 32**2, 96**2, float("inf")]

        labels = ["small", "medium", "large"]

        sizes = pd.cut(self.df["area"], bins=bins, labels=labels)

        return sizes.value_counts()

    def aspect_ratio_summary(self):

        return {
            "mean": self.df["aspect_ratio"].mean(),
            "median": self.df["aspect_ratio"].median(),
            "std": self.df["aspect_ratio"].std(),
        }
