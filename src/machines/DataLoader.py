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
    Data loader for reading machine data from a JSON file.

    Args:
        path (str): Path to the JSON file.
        columns (Optional[Dict[str, str]]): Optional mapping
        to indicate the column names in case
        they are not named timestamp and energy_value as expected
    """

    def __init__(self, path: str, columns: Optional[Dict[str, str]] = None):
        self.path = path
        self.columns = columns

    def load_datas(self) -> pd.DataFrame:
        """
        Load machine data from a JSON file
        and ensures the obtained data is structured as expected

        -Renames columns if necessary
        -Sets the timestamp column as index and sort by it
        -Validates the schema of the obtained DataFrame

        Returns:
            pd.DataFrame: The loaded and validated machine data.

        Raises:
            SchemaError if the DataFrame doesn't have
            the expected energy_value column
            and the Datetime column as index
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
