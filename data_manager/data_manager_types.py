import pydantic
from typing import Dict, List
from pydantic import BaseModel
from dataclasses import dataclass
import pandas as pd
import typing as t

class DataConfig(BaseModel):
    files: Dict[str, str]
    other_data: List[Dict[str, str]]


class AppConfig(BaseModel):
    base: str
    data_config: DataConfig


@dataclass
class ScanData:
    stat_overview: pd.DataFrame
    strategy_lookup: t.Dict
    entry_table: pd.DataFrame
    peak_table: pd.DataFrame


if __name__ == '__main__':
    # Generate the JSON schema from the AppConfig class
    app_config_schema = AppConfig.schema()

    # Validate an instance of the AppConfig class against the generated schema
    app_config = AppConfig.parse_obj({
        "base_dir": {"base": "C:\\Users\\bjahn\\OneDrive\\algo_data\\history\\sp500_1d_6_3"},
        "data_config": {
            "files": {"history": "history.csv", "bench": "bench.csv"},
            "other_data": [{"interval": "15m", "bench": "SPY"}]
        }
    })
    app_config_schema.validate(app_config.dict())
