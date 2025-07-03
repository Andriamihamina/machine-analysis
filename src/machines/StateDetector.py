import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, Generic, TypedDict, TypeVar

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclasses.dataclass
class StateDetectorConfig:
    pass


C = TypeVar("C", bound=StateDetectorConfig)


@dataclasses.dataclass
class KMeansStateDetectorConfig(StateDetectorConfig):
    n_clusters: int
    column_name: str


class ThresholdBoundaries(TypedDict, total=False):
    """
    Indicates the boundaries for a threshold state detector.

    :key lower: The lower boundary for the threshold.
    :type lower: int
    :key upper: The upper boundary for the threshold.
    :type upper: int
    """

    lower: int
    upper: int


@dataclasses.dataclass
class ThresholdStateDetectorConfig(StateDetectorConfig):
    """
    Configuration for a threshold-based state detector.

    :key column_name: The name of the column to apply the thresholds on.
    :type column_name: str
    :key thresholds: A dictionary mapping state names to their threshold boundaries.
    :type thresholds: Dict[str, ThresholdBoundaries]
    """

    column_name: str
    thresholds: Dict[str, ThresholdBoundaries]


class StateDetector(ABC, Generic[C]):

    @abstractmethod
    def detect_state(self, data: pd.DataFrame, config: C) -> pd.DataFrame:
        pass


class KMeansStateDetector(StateDetector[KMeansStateDetectorConfig]):

    def detect_state(
        self, data: pd.DataFrame, config: KMeansStateDetectorConfig
    ) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[config.column_name].to_frame())
        kmeans = KMeans(config.n_clusters, random_state=0)
        data["state"] = kmeans.fit_predict(scaled_data)
        return data


class ThresholdStateDetector(StateDetector[ThresholdStateDetectorConfig]):
    """
    Detects states based on predefined thresholds for a specific column.

    """

    def detect_state(
        self, data: pd.DataFrame, config: ThresholdStateDetectorConfig
    ) -> pd.DataFrame:
        """
        Detects states based on predefined thresholds for a specific column.

        :param data: Dataframe containing the energy consumption data
        :type data: pandas.DataFrame
        :param config: Configuration for the threshold state detector.
        :type config: ThresholdStateDetectorConfig
        :return: DataFrame with an additional 'state' column indicating the detected states.
        :rtype: pd.DataFrame
        """
        for state, boundaries in config.thresholds.items():
            mask = pd.Series(True, index=data.index)
            if "lower" in boundaries:
                lower = boundaries.get("lower")
                mask &= data[config.column_name] >= lower
            if "upper" in boundaries:
                upper = boundaries.get("upper")
                mask &= data[config.column_name] < upper

            data.loc[mask, "state"] = state
        return data
