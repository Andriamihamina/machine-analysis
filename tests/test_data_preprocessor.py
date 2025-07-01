import os
import tempfile

from src.machines.preprocessing import MachineDataPreprocessor
from tests.conftest import fake_datas_str


class TestDataPreprocessor:
    def setup_method(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+", suffix=".txt"
        )
        self.temp_file.write(fake_datas_str())
        self.temp_file.seek(0)

    def teardown_method(self) -> None:
        if self.temp_file:
            os.remove(self.temp_file.name)

    def test_load_json(self) -> None:
        df = MachineDataPreprocessor().load_json(self.temp_file.name)
        assert {"timestamp", "energy_value"}.issubset(df.columns)
