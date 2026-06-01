import pandas as pd


class SceneStats:

    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def weather_distribution(self):

        stats = (
            self.df["weather"]
            .value_counts()
            .reset_index()
        )

        stats.columns = [
            "weather",
            "count"
        ]

        return stats
    def timeofday_distribution(self):

        stats = (
            self.df["timeofday"]
            .value_counts()
            .reset_index()
        )

        stats.columns = [
            "timeofday",
            "count"
        ]

        return stats
    
    def scene_distribution(self):

        stats = (
            self.df["scene"]
            .value_counts()
            .reset_index()
        )

        stats.columns = [
            "scene",
            "count"
        ]

        return stats
    def summary(self):

        return {
            "weather_types":
                self.df["weather"].nunique(),

            "scene_types":
                self.df["scene"].nunique(),

            "timeofday_types":
                self.df["timeofday"].nunique(),
        }