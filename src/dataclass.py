from dataclasses import dataclass
from typing import List, Union

from numpy import ndarray


@dataclass
class LocationValue:
    geo_value: str
    geo_type: str
    date: int
    value: float


@dataclass
class LocationSeries:
    geo_value: str
    geo_type: str
    dates: List[int]
    values: Union[List, ndarray]
