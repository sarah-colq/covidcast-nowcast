from dataclasses import dataclass
from typing import List, Union

from numpy import ndarray


@dataclass(frozen=True)
class SignalConfig:
    source: str = None
    signal: str = None
    name: str = None
    model: str = None


@dataclass
class LocationSeries:
    geo_value: str = None
    geo_type: str = None
    dates: List[int] = None
    values: Union[List, ndarray] = None

    # add setter check enforcing that len(dates) == len(values)?