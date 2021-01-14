"""Run nowcast."""

from typing import List, Tuple

import numpy as np

from delphi_covidcast_nowcast.data_containers import SignalConfig
from delphi_covidcast_nowcast.deconvolution.deconvolution import deconvolve_signal
from delphi_covidcast_nowcast.nowcast_fusion import covariance, fusion
from delphi_covidcast_nowcast.sensorization.sensor import get_sensors


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
    train_sensors = get_sensors(input_dates[0], input_dates[-1],
                                sensor_indicators, ground_truth,
                                # ground_truth = None if compute_missing=False
                                compute_missing=True)  # change to false once we have DB

    now_sensors = get_sensors(nowcast_dates[0], nowcast_dates[0],
                              sensor_indicators, ground_truth,
                              compute_missing=True)

    ## put into matrix form
    # convert to dict indexed by loc to make matching across train/now easier
    y = dict((s.geo_value, s) for s in ground_truth)
    n_sensor_locs = len(sensor_indicators) * len(input_locations)
    noise = np.full((len(input_dates), n_sensor_locs), np.nan)
    z = np.full((1, n_sensor_locs), np.nan)
    j = 0
    for sensor in sensor_indicators:
        # convert to dict indexed by loc to make matching across train/now easier
        train_series = dict((s.geo_value, s) for s in train_sensors[sensor])
        now_series = dict((s.geo_value, s) for s in now_sensors[sensor])
        valid_sensor_locs = set(train_series.keys()) & set(now_series.keys())

        for loc in sorted(valid_sensor_locs):
            noise[:, j] = y[loc].values - train_series[loc].values
            z[:, j] = now_series[loc].values
            j += 1

    # cull nan columns
    finite_cols = np.logical_and(np.all(np.isfinite(noise), axis=0),
                                 np.all(np.isfinite(z), axis=0))
    noise = noise[:, finite_cols]
    z = z[:, finite_cols]

    print(noise.shape, z.shape)

    # generate statespace
    # to do
    H = np.ones((noise.shape[1], len(ground_truth))) / len(ground_truth)
    H[0, 0] = 1
    W = np.ones((len(ground_truth), len(ground_truth))) / len(ground_truth)

    # estimate covariance
    R = covariance.mle_cov(noise, covariance.BlendDiagonal2)

    # run SF
    x, P = fusion.fuse(z, R, H)
    y, S = fusion.extract(x, P, W)
    stdev = np.sqrt(np.diag(S)).reshape(y.shape)

    # return output
    pass
