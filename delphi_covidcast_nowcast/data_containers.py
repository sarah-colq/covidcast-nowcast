from dataclasses import dataclass, field
from typing import List, Union

from numpy import ndarray


@dataclass(frozen=True)
class SignalConfig:
    source: str = None
    signal: str = None
    name: str = None


@dataclass(frozen=True)
class LocationSeries:
    geo_value: str = None
    geo_type: str = None
    dates: List[int] = field(default_factory=lambda: [])
    values: Union[List, ndarray] = field(default_factory=lambda: [])

    def __post_init__(self):
        if (self.dates is None and self.values is not None) or \
                (self.dates is not None and self.values is None):
            raise ValueError("Must init with both date and values")

        if (self.dates is not None and self.values is not None) and \
                (len(self.dates) != len(self.values)):
            raise ValueError("Length of dates and values differs")

    def add_data(self, date, value):
        """Append a date and value to existing attributes.

        Safer than appending individually since the two lists can't have different lengths
        """
        object.__setattr__(self, 'dates', self.dates + [date])
        object.__setattr__(self, 'values', self.values + [value])
