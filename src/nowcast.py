"""Run nowcast."""

from typing import List, Tuple

import numpy as np


def nowcast(input_dates: List[int],
            input_location: List[Tuple[str, str]],
            sensor_indicators: List[Tuple[str, str]],
            ground_truth_indicator: Tuple[str, str],
            kernel: List[float],
            nowcast_dates: List[int] = "*",
            ) -> Tuple[np.ndarray, np.ndarray, List]:
    """

    Parameters
    ----------
    input_dates
        List of dates to train data on and get nowcasts for.
    input_location
        List of (location, geo_type) tuples specifying locations to train and obtain nowcasts for.
    sensor_indicators
        List of (source, signal) tuples specifying indicators to use as sensors.
    ground_truth_indicator
        (source, signal) tuple of quantity to nowcast.
    kernel
        Delay distribution to deconvolv with ground_truth_indicator
    nowcast_dates
        Dates to get predictions for. Defaults to input_dates + additional day.

    Returns
    -------
        (predicted values, std devs, locations)
    """
    # get geo mappings
    # deconvolve for ground truth
    # fit sensors
    # generate statespace
    # estimate covariance
    # run SF
    # return output
    pass


