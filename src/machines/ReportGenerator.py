import pandas as pd

from .AnomalyDetector import AnomalyDetector


class ReportGenerator:
    """
    Generates reports of the active production cycles.
    """

    def count_units(
        self, data: pd.DataFrame, start_time: str, end_time: str
    ) -> int:
        """
        Count the number of cycles overlapping with the time window.


        :param data: dataframe containing the discrete production cycles
        :type data: pd.DataFrame
        :param start_time: start time of the time window
        :type start_time: str
        :param end_time: end time of the time window
        :type end_time: str
        :return: the number of cycles overlapping with the time window
        :rtype: int
        """
        return len(data[pd.to_datetime(start_time) : pd.to_datetime(end_time)])

    def flag_anomalies(
        self, data: pd.DataFrame, detector: AnomalyDetector
    ) -> pd.DataFrame:
        """
        Detect anomalies in the data using the provided anomaly detector.

        :param data: _description_
        :type data: pd.DataFrame
        :param detector:
            Anomaly detector instance that implements the detect method.
        :type detector: AnomalyDetector
        :return: DataFrame with an additional 'anomaly'
        boolean column indicating anomalies.
        :rtype: pd.DataFrame
        """
        return detector.detect(data)
