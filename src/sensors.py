"""Functions to run sensorization."""

from typing import List, Tuple

import numpy as np

from src.ar_sensor import ArSensor


def generate_sensors(input_dates: List[int],
                     input_location: List[Tuple[str, str]],
                     sensor_indicators: List[Tuple[str, str]],
                     ) -> List[Tuple]:
    """

    Parameters
    ----------
    input_dates
        List of dates to train data on and get nowcasts for.
    input_location
        List of (location, geo_type) tuples specifying locations to train and obtain nowcasts for.
    sensor_indicators
        List of (source, signal) tuples specifying indicators to use as sensors.

    Returns
    -------
        TBD
    """
    # api_sensors = []
    # ar_sensors = []
    # for each location:
    #     ar_sensors.append(get_ar_sensor(location, values, ...))
    #     for each indicator:
    #         sensors.append(get_indicator_sensor(location, indicator, input_dates)
    # return api_sensors + ar_sensors
    pass


def get_ar_sensor(input_dates: List[int], signal: np.ndarray,
                  n_lags: int = 3) -> Tuple[List[int], np.ndarray]:
    """
    The n_lags points in input_dates/signal must be the most recent dates to the
    desired nowcast date! Otherwise, we cannot predict the point at nowcast date.

    Parameters
    ----------
    input_dates
    signal
    n_lags

    Returns
    -------

    """
    # get existing sensor values from DB
    # for dates that arent available (hopefully just today), grab necessary window of indicator values
    # return sensor values up to and including today (current sensor value)

    sensors = []  # todo: get from DB

    # create an auto-regression "sensor" for all locations
    # 3 covariates, no intercept, small L2 penalty
    B, Yhat, dates, z = ArSensor.get_sensor(input_dates, signal, n_lags, False, 0.1)

    # todo: insert z into db
    # temp hack, just return the in-sample values
    sensors = Yhat

    return dates, Yhat, z


def get_indicator_sensor():
    # same as ar_sensor but with different signals
    pass
