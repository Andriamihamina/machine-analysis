from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame


class MachineDataLoader(ABC):
    """Abstract base class for machine data loaders."""

    @abstractmethod
    def load_datas(self) -> pd.DataFrame:
        """Load machine data from different sources into a pandas DataFrame.

        Returns:
            pd.DataFrame
        """
        pass


class JsonDataLoader(MachineDataLoader):
    """
    Loader that loads machine data from a JSON file.
    """

    def __init__(self, path: str, columns: Optional[Dict[str, str]] = None):
        """
        Initializes the JsonDataLoader with a file path
        and optional column renaming.

        :param path: Path to the JSON file containing machine data.
        :type path: str
        :param columns:
            Mapping of columns in case names need to be renamed, defaults to None
        :type columns: Optional[Dict[str, str]], optional
        """
        self.path = path
        self.columns = columns

    def load_datas(self) -> pd.DataFrame:
        """
        Proceeds to load the data from the JSON file,
        renaming columns if specified, and validating the schema.

        :raises:
            SchemaError if the DataFrame does not conform to the expected schema.
        :return:
            DataFrame containing the machine data with 'timestamp' as index.
        :rtype: pandas.DataFrame
        """
        df: DataFrame[Any] = pd.read_json(self.path)
        if self.columns:
            df.rename(columns=self.columns, inplace=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df.dropna()

        schema = pa.DataFrameSchema(
            columns={
                "energy_value": pa.Column(
                    pa.Float64, required=True, nullable=True
                )
            },
            index=pa.Index(pa.DateTime, nullable=True),
            coerce=True,
        )

        try:
            schema.validate(df)
            return df
        except SchemaError as e:
            raise e
