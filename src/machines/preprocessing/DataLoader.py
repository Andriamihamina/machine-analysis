from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame


class MachineDataLoader(ABC):

    @abstractmethod
    def load_datas(self) -> DataFrame[Any]:
        pass


class JsonDataLoader(MachineDataLoader):

    def __init__(self, path: str, columns: Optional[Dict[str, str]] = None):
        self.path = path
        self.columns = columns

    def load_datas(self) -> DataFrame[Any]:
        df: DataFrame[Any] = pd.read_json(self.path)
        if self.columns:
            df.rename(columns=self.columns, inplace=True)
        df.set_index(df["timestamp"], inplace=True)
        df.sort_index(inplace=True)

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
