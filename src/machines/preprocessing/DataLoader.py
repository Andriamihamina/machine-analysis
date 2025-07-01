from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
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
        return df
