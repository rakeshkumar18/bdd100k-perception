import pandas as pd


class OcclusionStats:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overall_occlusion_rate(self):

        total = len(self.df)

        occluded = self.df["occluded"].sum()

        return {
            "total_objects": total,
            "occluded_objects": int(occluded),
            "occlusion_rate": round(100 * occluded / total, 2),
        }

    def occlusion_by_class(self):

        stats = self.df.groupby("category")["occluded"].mean() * 100

        return stats.sort_values(ascending=False).reset_index()

    def truncation_rate(self):

        total = len(self.df)

        truncated = self.df["truncated"].sum()

        return {
            "truncated_objects": int(truncated),
            "truncation_rate": round(100 * truncated / total, 2),
        }
