from dataclasses import dataclass
from typing import List, Union

from numpy import nan, ndarray


@dataclass
class LocationPoint:
    geo_value: str = None
    geo_type: str = None
    date: int = None
    value: float = nan


@dataclass
class LocationSeries:
    geo_value: str = None
    geo_type: str = None
    dates: List[int] = list
    values: Union[List, ndarray] = list
