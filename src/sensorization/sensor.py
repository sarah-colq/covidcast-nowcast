"""Functions to run sensorization."""

from typing import List, Tuple, Dict

import numpy as np
import covidcast
from delphi_epidata import Epidata

from ..data_containers import LocationSeries
from .model import compute_ar_sensor, compute_regression_sensor


def get_sensors(start_date: int,
                end_date: int,
                sensors: List[Tuple[str, str, str, str]],
                ground_truths: List[LocationSeries] = None,
                compute_missing: bool = False,
                ) -> Dict[Tuple[str, str, str, str], LocationSeries]:
    """
    Return sensorized values from start to end date at given locations for specified sensors.

    If compute_missing is True, we attempt to recompute values which cannot be retrieved from
    the Epidata db based on most recent covidcast data.

    Only locations that have complete ground truths (no nans) will have sensorization values
    retrieved or computed, even if compute_missing=False and the ground truth is not needed. This
    behavior should probably be updated.

    Parameters
    ----------
    start_date
        first day to attempt to get sensor values for.
    end_date
        last day to attempt to get sensor values for.
    sensors
        list of tuples specifying (source, signal, sensor_name, model) for sensors
    ground_truths
        list of LocationSeries, one for each location desired. If `compute_missing=False`, ground
        truth is not needed because no training is occuring, and this argument is ignored.
    compute_missing
        boolean specifying whether the function should attempt to compute any dates which
        were not retrieved from historical data. Defaults to False.

    Returns
    -------
        Dict where keys are sensor tuples and values are lists, where each list element is a
        LocationSeries holding sensor data for a location.
    """
    output = {}
    locations_to_get = [y for y in ground_truths if not np.all(np.isnan(y.values))]
    unavail_loc = [y for y in ground_truths if np.all(np.isnan(y.values))]  # need to clean this up
    if unavail_loc:
        print(f"unavailable locations {unavail_loc}")

    for location in locations_to_get:
        # should come up with an easier way to do this too.
        ground_truth_value = [] if not compute_missing else\
            [i for i in ground_truths if (i.geo_value, i.geo_type) == location][0]
        for sensor in sensors:
            output[sensor] = get_sensor_values(sensor, location, start_date, end_date,
                                               ground_truth_value, compute_missing)[0]

    return output


def get_sensor_values(sensor: Tuple[str, str, str, str],
                      location: Tuple[str, str],
                      start_date: int,
                      end_date: int,
                      ground_truth: LocationSeries,
                      compute_missing: bool) -> Tuple[LocationSeries, list]:
    """
    Return sensorized values for a single location, using available historical data if specified.
    
    Parameters
    ----------
    sensor
        (source, signal, sensor_name, model) tuple specifying which sensor to retrieve/compute.
    location
        (geo_value, geo_type) tuple specifying where to get values for.
    start_date
        first day to attempt to get sensor values for.
    end_date
        last day to attempt to get sensor values for.
    ground_truth
        LocationSeries containing ground truth values to train against. Ignored if
        compute_missing=False
    compute_missing
        Flag for whether or not missing values should be recomputed.

    Returns
    -------
        Tuple of (LocationSeries of sensor data, dates where no values were obtained).
    """
    # left out recompute_all_data argument for now just to keep things simple
    output, missing_dates = _get_historical_data(sensor, start_date, end_date)
    if not compute_missing or not missing_dates:
        return output, missing_dates
    indicator_values = covidcast.signal(  # gets all available data for now, could be optimized
        sensor[0], sensor[1], geo_values=location[0], geo_type=location[1]
    )  # reformat indicator_values to value/day pairs or a LocationSeries?
    for date in missing_dates:
        if sensor[3] == "ar":
            sensor_value = compute_ar_sensor(date, indicator_values)
        elif sensor[3] == "regression":
            sensor_value = compute_regression_sensor(date, indicator_values, ground_truth)
        else:
            raise ValueError("Invalid sensorization method. Must be 'ar' or 'regression'")
        output.values.append(sensor_value)  # what if its a numpy array? would need to change method
        output.dates.append(date)
        _export_to_csv(sensor_value)
    return output, missing_dates


def _get_historical_data(indicator, min_date, max_date) -> Tuple[LocationSeries, list]:
    """Query Epidata API for historical sensorization data."""
    Epidata.covidcast_nowcast(source=indicator[0], signal=indicator[1], sensor_name=indicator[3])
    # convert data to proper format (list of(value, date) tuples?)
    # compute missing dates between min_date and max date
    # return data, missing_dates
    pass


def _export_to_csv(value):
    """Save value to csv for upload to epidata database"""
    pass

