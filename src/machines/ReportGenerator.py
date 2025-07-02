import pandas as pd

from .AnomalyDetector import AnomalyDetector


class ReportGenerator:
    def production_in_time(
        self, data: pd.DataFrame, start_time: str, end_time: str
    ) -> pd.DataFrame:
        """
        Return production cycles that overlap with the given time window.
        """
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        mask = (pd.to_datetime(data["start_time"]) <= end_time) & (
            pd.to_datetime(data["end_time"]) > start_time
        )
        return data.loc[mask].reset_index(drop=True)

    def count_units(
        self, data: pd.DataFrame, start_time: str, end_time: str
    ) -> int:
        """
        Count the number of cycles overlapping with the time window.
        """

        subset = self.production_in_time(data, start_time, end_time)
        return len(subset)

    def flag_anomalies(
        self, data: pd.DataFrame, detector: AnomalyDetector
    ) -> pd.DataFrame:
        return detector.detect(data)
