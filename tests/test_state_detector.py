import pandas as pd

from src.machines.StateDetector import (
    ThresholdStateDetector,
    ThresholdStateDetectorConfig,
)
from tests.conftest import fake_datas_df


class TestStateDetector:

    def setup_method(self) -> None:
        self.datas = pd.DataFrame(
            [
                {
                    "timestamp": "2025-05-11T07:08:49.154000",
                    "energy_value": 25.96,
                },
                {
                    "timestamp": "2025-05-11T07:08:49.155000",
                    "energy_value": 300,
                },
                {"timestamp": "2025-05-11T07:08:49.156000", "energy_value": 0},
                {
                    "timestamp": "2025-05-11T07:08:49.413000",
                    "energy_value": -300,
                },
            ]
        )

    def test_threshold_no_boundaries(self) -> None:
        config = ThresholdStateDetectorConfig(
            column_name="energy_value", thresholds={"active": {}}
        )

        datas = fake_datas_df()
        detector = ThresholdStateDetector()
        df = detector.detect_state(datas, config)
        assert "state" in df.columns
        assert df["state"].nunique() == 1

    def test_threshold_detection(self) -> None:

        config = ThresholdStateDetectorConfig(
            column_name="energy_value",
            thresholds={
                "anomaly": {
                    "upper": 0,
                },
                "idle": {"lower": 0, "upper": 30},
                "active": {"lower": 30},
            },
        )

        detector = ThresholdStateDetector()
        df = detector.detect_state(self.datas, config)

        assert "state" in df.columns
        assert len(df.loc[df["state"] == "anomaly"]) == 1
        assert len(df.loc[df["state"] == "idle"]) == 2
        assert len(df.loc[df["state"] == "active"]) == 1
