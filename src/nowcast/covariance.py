"""Estimate the covariance matrix."""

from typing import List, Tuple

import numpy as np

def estimate_covariance(input_dates: List[int],
                        sensors: List[Tuple],
                        ground_truth_values: np.ndarray):
    """
    Estimate the sensor noise covariance matrix R.


    Parameters
    ----------
    input_dates

    sensors
    ground_truth_values

    Returns
    -------

    """
    # basically wrap https://github.com/dfarrow0/covidcast-nowcast/blob/0de33880368eac3df6c1ac7b2d4ba6a41ff85480/src/sf/nowcast.py#L239-L252
    # R = covariance.mle_cov(noise, covariance.BlendDiagonal2)
    # return R
    pass
