from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector(ABC):

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the provided data and add an 'anomaly' column.

        :param data: dataframe containing the energy consumption data
        :type data: pandas.DataFrame
        :return:
            DataFrame with an additional 'anomaly' boolean column indicating anomalies
        :rtype: pandas.DataFrame
        """
        pass


class ThresholdAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies based on energy consumption thresholds.
    """

    def __init__(
        self,
        mean_energy_theshold: float = 50.0,
        peak_energy_threshold: float = 150.0,
    ):
        """
        Constructor for the ThresholdAnomalyDetector.

        :param mean_energy_theshold:
        threshold above which the cycle is flagged as anomaly,
        defaults to 50.0
        :type mean_energy_theshold: float, optional
        :param peak_energy_threshold:
        the cycle is flagged as an anomaly if the energy exceeds
        that value during the cycle , defaults to 150.0
        :type peak_energy_threshold: float, optional
        """
        self.mean_energy_threshold = mean_energy_theshold
        self.peak_energy_threshold = peak_energy_threshold

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an boolean column 'anomaly'  to the dataframe indicating anomaly


        :param data: Dataframe containing discrete production cycles
        :type data: pandas.DataFrame
        :return: DataFrame with an additional 'anomaly'
        boolean column indicating anomalies
        :rtype: pandas.DataFrame
        """
        data = data.copy()
        conditions = (data["mean_energy"] > self.mean_energy_threshold) | (
            data["peak_energy"] > self.peak_energy_threshold
        )

        data["anomaly"] = conditions
        return data


class IsolationForestAnomalyDetector(AnomalyDetector):
    """
    Detects anomalies based on energy consumption thresholds.
    """

    def __init__(self, config: Dict[Any, Any]):
        """
        Config for the IsolationForestAnomalyDetector.

        :param config: Isolation Forest configuration parameters.
        :type config: Dict[Any, Any]
        """
        self.config = config

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an boolean column 'anomaly'  to the dataframe indicating anomaly


        :param data: Dataframe containing discrete production cycles
        :type data: pandas.DataFrame
        :return: DataFrame with an additional 'anomaly'
        boolean column indicating anomalies
        :rtype: pandas.DataFrame
        """
        data = data.copy()
        iso_forest = IsolationForest(self.config)
        data["anomaly"] = iso_forest.fit_predict(data)

        return data
