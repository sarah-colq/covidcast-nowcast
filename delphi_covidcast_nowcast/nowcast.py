"""Run nowcast."""

from typing import List, Tuple

import numpy as np

from .deconvolution.deconvolution import deconvolve_signal
from .data_containers import SignalConfig

def nowcast(input_dates: List[int],
            input_locations: List[Tuple[str, str]],
            sensor_indicators: List[SignalConfig],
            convolved_truth_indicator: SignalConfig,
            kernel: List[float],
            nowcast_dates: List[int] = "*",
            ) -> Tuple[np.ndarray, np.ndarray, List]:
    """

    Parameters
    ----------
    input_dates
        List of dates to train data on and get nowcasts for.
    input_locations
        List of (location, geo_type) tuples specifying locations to train and obtain nowcasts for.
    sensor_indicators
        List of (source, signal) tuples specifying indicators to use as sensors.
    convolved_truth_indicator
        (source, signal) tuple of quantity to deconvolve.
    kernel
        Delay distribution to deconvolve with convolved_truth_indicator
    nowcast_dates
        Dates to get predictions for. Defaults to input_dates + additional day.

    Returns
    -------
        (predicted values, std devs, locations)
    """
    # get geo mappings

    # deconvolve for ground truth
    ground_truth = deconvolve_signal(convolved_truth_indicator, input_dates,
                                     input_locations, np.array(kernel))

    # fit sensors
    # generate statespace
    # estimate covariance
    # run SF
    # return output
    pass
