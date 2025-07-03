import tempfile

from src.machines.DataLoader import JsonDataLoader
from tests.conftest import fake_datas_str


class TestJsonLoader:
    def setup_method(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+", suffix=".txt"
        )
        self.temp_file.write(fake_datas_str())
        self.temp_file.seek(0)

    def test_load_json(self) -> None:
        loader = JsonDataLoader()
        df = loader.load(
            self.temp_file.name, {"value": "energy_value", "time": "timestamp"}
        )
        assert not df.empty
        assert "energy_value" in df.columns
        assert df.index.name == "timestamp"

    def test_load_json_columns_not_matching(self) -> None:
        loader = JsonDataLoader()
        df = loader.load(self.temp_file.name)
        assert not df.empty
        assert "energy_value" in df.columns
        assert df.index.name == "timestamp"
