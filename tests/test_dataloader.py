import io
import json
import os
import tempfile

import pandas as pd
import pytest

from src.machines.preprocessing.DataLoader import JsonDataLoader
from tests.conftest import fake_datas_str


class TestJsonLoader:
    def setup_method(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+", suffix=".txt"
        )
        self.temp_file.write(fake_datas_str())
        self.temp_file.seek(0)

    def teardown_method(self) -> None:
        try:
            os.remove(self.temp_file.name)
        except:
            pass

    def test_load_json(self) -> None:
        loader = JsonDataLoader(
            self.temp_file.name, {"value": "energy_value", "time": "timestamp"}
        )
        df = loader.load_datas()
        assert not df.empty
        assert {"timestamp", "energy_value"}.issubset(df.columns)
        assert df.index.name == "timestamp"

    def test_load_json_columns_not_matching(self) -> None:
        loader = JsonDataLoader(self.temp_file.name)
        df = loader.load_datas()
        assert not df.empty
        assert {"timestamp", "energy_value"}.issubset(df.columns)
        assert df.index.name == "timestamp"
