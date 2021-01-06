"""Functions to run sensorization."""

from typing import List, Tuple

import numpy as np

from src.ar_sensor import ArSensor
from src.dataclass import LocationPoint, LocationSeries


def generate_sensors(input_dates: List[int],
                     input_locations: List[Tuple[str, str]],
                     sensor_indicators: List[Tuple[str, str]],
                     ground_truth: List[LocationSeries],
                     nowcast_date: int,
                     ) -> Tuple[List[LocationSeries], List[LocationPoint]]:
    """

    Parameters
    ----------
    input_dates
        List of dates to train data on.
    input_locations
        List of (location, geo_type) tuples specifying locations to train and obtain nowcasts for.
    sensor_indicators
        List of (source, signal) tuples specifying indicators to use as sensors.
    ground_truth
        List of LocationSeries containing deconvolved ground truth to train response.
    nowcast_date
        Date to get prediction for.

    Returns
    -------
        TBD
    """
    # first get metadata on which locations we have responses
    avail_truth = [y for y in ground_truth if not np.all(np.isnan(y.values))]
    avail_loc = [y.geo_value for y in avail_truth]
    unavail_loc = [x[0] for x in input_locations if x[0] not in avail_loc]

    if len(unavail_loc) > 0:
        print(f"unavailable locations {unavail_loc}")

    # by now we only generate sensors for locations in avail_loc

    # ar sensor
    ar_sensors_train = []
    ar_sensors_test = []
    for signal in avail_truth:
        ar_train, ar_test = get_ar_sensor(signal.dates, signal.values)
        ar_sensors_train.append(
            LocationSeries(signal.geo_value, signal.geo_type,
                           ar_train.dates, ar_train.values))
        ar_sensors_test.append(
            LocationPoint(signal.geo_value, signal.geo_type,
                          ar_test.date, ar_test.value))

    # api_sensors = []
    # ar_sensors = []
    # for each location:
    #     ar_sensors.append(get_ar_sensor(location, values, ...))
    #     for each indicator:
    #         sensors.append(get_indicator_sensor(location, indicator, input_dates)
    # return api_sensors + ar_sensors
    return ar_sensors_train, ar_sensors_test


def get_ar_sensor(input_dates: List[int],
                  signal: np.ndarray,
                  n_lags: int = 3) -> Tuple[LocationSeries, LocationPoint]:
    """
    The n_lags points in input_dates/signal must be the most recent dates to the
    desired nowcast date! Otherwise, we cannot predict the point at nowcast date.

    Parameters
    ----------
    input_dates
        training dates
    signal
    n_lags

    Returns
    -------
        tuple of historical sensor values, and current sensor value (one-ahead)

    """
    # get existing sensor values from DB
    # for dates that arent available (hopefully just today), grab necessary window of indicator values
    # return sensor values up to and including today (current sensor value)
    sensors = []  # todo: get from DB

    # create an auto-regression "sensor" for all locations
    # 3 covariates, no intercept, small L2 penalty
    # todo: check ArSensor code to make sure the dates are as expected
    B, Yhat, dates, z = ArSensor.get_sensor(input_dates, signal, n_lags, False, 0.1)

    # todo: insert z into db
    # temp hack, just return the in-sample values
    sensors = Yhat

    return LocationSeries(dates=dates[:-1], values=sensors), \
           LocationPoint(date=dates[-1], value=z)


def get_api_sensor(input_dates: List[int],
                   signal: np.ndarray):
    # get existing sensor values from DB
    # for dates that arent available (hopefully just today), grab necessary window of indicator values
    # return sensor values up to and including today (current sensor value)
    sensors = []  # todo: get from DB
    pass
