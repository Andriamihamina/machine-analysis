import dataclasses
import functools
import inspect
import json
import typing
import warnings
from abc import ABC
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

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
