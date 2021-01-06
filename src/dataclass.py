from dataclasses import dataclass
from typing import List, Union

import numpy as np


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
    values: Union[List, np.ndarray]
