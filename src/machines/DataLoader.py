from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame


class MachineDataLoader(ABC):
    """Abstract base class for machine data loaders."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load machine data from different sources into a pandas DataFrame.

        Returns:
            pd.DataFrame
        """
        pass


class JsonDataLoader(MachineDataLoader):
    """
    Loader that loads machine data from a JSON file.
    """

    def load(
        self, path: str, column_mappings: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Proceeds to load the data from the JSON file,
        renaming columns if specified, and validating the schema.

        :param path: Path to the JSON file containing machine data.
        :type path: str
        :param column_mappings: Mapping of columns in case names need to be renamed, defaults to None
        :type column_mappings: Optional[Dict[str, str]], optional
        :raises: SchemaError if the DataFrame does not conform to the expected schema.
        :return: DataFrame containing the machine data with 'timestamp' as index.
        :rtype: pandas.DataFrame
        """
        df: DataFrame[Any] = pd.read_json(path)
        if column_mappings:
            df.rename(columns=column_mappings, inplace=True)
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
