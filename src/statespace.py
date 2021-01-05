"""Estimate the covariance matrix."""

from typing import List, Tuple

import numpy as np


def generate_statespace(sensors: List[Tuple],
                        input_locations: List[Tuple],
                        geos: Tuple
                        ) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Generate W and H matrices.

    Parameters
    ----------
    sensors
        TBD exactly, but contains sensors values and dates
    input_locations
        Locations to train and predict nowcasts for.
    geos
        Tuple of (county list, msa-county mapping, state-county mapping).

    Returns
    -------
        Full rank matrices W and H
    """
    # num_sensors = len(sensors)
    # num_locs = len(geos[0])
    # W0 = generate_W0(geos, num_sensors)
    # H0 = generate_H0(geos, num_locs)
    # return determine_statespace(W0, H0)
    pass


def generate_H0(geos: Tuple,
                num_sensors: int) -> np.ndarray:
    """
    Build matrix of all available sensor-location pairs.

    Parameters
    ----------
    geos
        Tuple of (county list, msa-county mapping, state-county mapping).
    num_sensors
        Number of sensors being fused.

    Returns
    -------
        Matrix of dimension # nowcasts X # counties. Usually rank deficient.
    """
    # Wrap this code https://github.com/dfarrow0/covidcast-nowcast/blob/0de33880368eac3df6c1ac7b2d4ba6a41ff85480/src/sf/nowcast.py#L176-L192
    pass


def generate_W0(geos: Tuple,
                num_locations: int) -> np.ndarray:
    """
    Build matrix determining all locations we want a nowcast.

    Parameters
    ----------
    geos
        Tuple of (county list, msa-county mapping, state-county mapping).
    num_locations
        Number of locations, which are counties in our case.

    Returns
    -------
        Matrix of dimension # sensors X # counties. Usually rank deficient.
    """
    # Wrap this code https://github.com/dfarrow0/covidcast-nowcast/blob/0de33880368eac3df6c1ac7b2d4ba6a41ff85480/src/sf/nowcast.py#L200-L215
    pass
