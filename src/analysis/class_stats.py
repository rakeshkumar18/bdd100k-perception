import pandas as pd


class ClassStats:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def class_counts(self) -> pd.DataFrame:
        """
        Count object instances per class
        """

        counts = self.df["category"].value_counts().reset_index()

        counts.columns = ["category", "count"]

        return counts

    def class_distribution(self):

        counts = self.class_counts()

        total = counts["count"].sum()

        counts["percentage"] = counts["count"] / total * 100

        return counts

    def top_classes(self, n=10):

        return self.class_distribution().head(n)

    def rare_classes(self, threshold=0.5):
        """
        Classes representing less than threshold %
        """

        stats = self.class_distribution()

        return stats[stats["percentage"] < threshold]

    def imbalance_ratio(self):

        counts = self.class_counts()["count"]

        return counts.max() / counts.min()

    def summary(self):

        stats = self.class_distribution()

        return {
            "num_classes": len(stats),
            "num_objects": int(stats["count"].sum()),
            "largest_class": stats.iloc[0]["category"],
            "largest_count": int(stats.iloc[0]["count"]),
            "smallest_class": stats.iloc[-1]["category"],
            "smallest_count": int(stats.iloc[-1]["count"]),
            "imbalance_ratio": self.imbalance_ratio(),
        }
