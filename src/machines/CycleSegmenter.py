from abc import ABC, abstractmethod

import pandas as pd


class CycleSegmenter(ABC):

    @abstractmethod
    def segment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an boolean 'anomaly' column to the dataframe indicating anomaly
        """
        pass


class StateCycleSegmenter(CycleSegmenter):

    def segment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Segment cycles based on state changes.
            Args:
                data (DataFrame): the data frame to segment.
        """
        data["cycle_change"] = data["state"].ne(data["state"].shift())
        data["cycle_id"] = data["cycle_change"].cumsum()
        cycle_groups = data[data["state"] == "active"].groupby("cycle_id")

        return cycle_groups.agg(
            start_time=("energy_value", lambda x: x.index.min()),
            end_time=("energy_value", lambda x: x.index.max()),
            duration_s=(
                "energy_value",
                lambda x: (x.index.max() - x.index.min()).total_seconds(),
            ),
            total_consumption=("energy_value", "sum"),
            peak_energy=("energy_value", "max"),
            mean_energy=("energy_value", "mean"),
            std_energy=("energy_value", "mean"),
        ).reset_index()
