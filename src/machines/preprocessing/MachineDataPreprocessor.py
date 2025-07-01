import pandas as pd

from .DataLoader import JsonDataLoader

# class PipelineConfig(ABC):
# near_constant_features_config: typing.Optional[NearConstantFeaturesConfig]


# AnyConfig = TypeVar("AnyConfig")
# Transformation = typing.Callable[[pd.DataFrame, AnyConfig], pd.DataFrame]
# TransformationLifted = typing.Callable[
#     [pd.DataFrame, PipelineConfig], pd.DataFrame
# ]


class MachineDataPreprocessor:
    def load_json(self, path: str) -> pd.DataFrame:
        df = JsonDataLoader(path).load_datas()
        return df
