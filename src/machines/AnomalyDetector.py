from abc import ABC, abstractmethod

import pandas as pd


class AnomalyDetector(ABC):

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an boolean column 'anomaly' to the dataframe indicating anomaly
        """
        pass


class ThresholdAnomalyDetector(AnomalyDetector):

    def __init__(
        self,
        mean_energy_theshold: float = 50.0,
        peak_energy_threshold: float = 150.0,
    ):
        self.mean_energy_threshold = mean_energy_theshold
        self.peak_energy_threshold = peak_energy_threshold

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an boolean column 'anomaly'  to the dataframe indicating anomaly
        """
        data = data.copy()
        conditions = (data["mean_energy"] > self.mean_energy_threshold) | (
            data["peak_energy"] > self.peak_energy_threshold
        )

        data["anomaly"] = conditions
        return data
