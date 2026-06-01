import pandas as pd


class TrainValComparison:

    def __init__(
        self,
        train_df,
        val_df
    ):
        self.train_df = train_df
        self.val_df = val_df

    def compare_classes(self):

        train_counts = (
            self.train_df["category"]
            .value_counts(normalize=True)
            * 100
        )

        val_counts = (
            self.val_df["category"]
            .value_counts(normalize=True)
            * 100
        )

        comparison = pd.concat(
            [train_counts, val_counts],
            axis=1
        )

        comparison.columns = [
            "train_pct",
            "val_pct"
        ]

        comparison = comparison.fillna(0)

        return comparison.reset_index()
    def compare_column(
    self,
    column
    ):

        train_dist = (
            self.train_df[column]
            .value_counts(normalize=True)
            * 100
        )

        val_dist = (
            self.val_df[column]
            .value_counts(normalize=True)
            * 100
        )

        comparison = pd.concat(
            [train_dist, val_dist],
            axis=1
        )

        comparison.columns = [
            "train_pct",
            "val_pct"
        ]

        comparison = (
        comparison
        .fillna(0)
        .reset_index()
        )

        comparison.rename(
            columns={
                "index": column
            },
            inplace=True
        )

        return comparison