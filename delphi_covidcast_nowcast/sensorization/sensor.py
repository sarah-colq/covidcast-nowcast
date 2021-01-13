"""Functions to run sensorization."""
import os
from typing import List, Tuple, Dict
from datetime import date
import numpy as np
from pandas import date_range
from delphi_epidata import Epidata

from ..data_containers import LocationSeries, SignalConfig
from .ar_model import compute_ar_sensor
from .regression_model import compute_regression_sensor


def get_sensors(start_date: int,
                end_date: int,
                sensors: List[SignalConfig],
                ground_truths: List[LocationSeries] = None,
                compute_missing: bool = False,
                ) -> Dict[SignalConfig, LocationSeries]:
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
    locations_to_get = [y for y in ground_truths if not np.any(np.isnan(y.values))]
    unavail_loc = [y for y in ground_truths if np.any(np.isnan(y.values))]  # need to clean this up
    if unavail_loc:
        print(f"unavailable locations {unavail_loc}")
    for location_truth in locations_to_get:
        output["ground_truth_ar"] = output.get("ground_truth_ar", []) + [get_ar_sensor_values(
            location_truth, start_date, end_date
        )]
        for sensor in sensors:
            output[sensor] = output.get(sensor, []) + [get_regression_sensor_values(
                sensor, start_date, end_date, location_truth, compute_missing
            )]
    return output


def get_ar_sensor_values(values: LocationSeries,
                         start_date: int,
                         end_date: int) -> LocationSeries:
    """

    Parameters
    ----------
    values
    start_date
    end_date

    Returns
    -------

    """
    output = LocationSeries(values.geo_value, values.geo_type, [], [])
    for day in [int(i.strftime("%Y%m%d")) for i in date_range(str(start_date), str(end_date))]:
        sensor_value = compute_ar_sensor(day, values)
        if np.isnan(sensor_value):
            continue
        output.values.append(sensor_value)  # if np array would need to change append method
        output.dates.append(day)
    return output


def get_regression_sensor_values(sensor: SignalConfig,
                                 start_date: int,
                                 end_date: int,
                                 ground_truth: LocationSeries,
                                 compute_missing: bool) -> LocationSeries:
    """
    Return sensorized values for a single location, using available historical data if specified.

    If new values are to be computed, they currently are done with the most recent issue of data,
    as opposed to the data available as_of the desired date.
    
    Parameters
    ----------
    sensor
        (source, signal, sensor_name, model) tuple specifying which sensor to retrieve/compute.
    start_date
        first day to attempt to get sensor values for.
    end_date
        last day to attempt to get sensor values for.
    ground_truth
        LocationSeries containing ground truth values to train against. Also used to transfer geo
        information. Values are ignored if compute_missing=False
    compute_missing
        Flag for whether or not missing values should be recomputed.

    Returns
    -------
        LocationSeries of sensor data.
    """
    # left out recompute_all_data argument for now just to keep things simple

    ############### COMMENT OUT FOR TESTING ###############
    output, missing_dates = _get_historical_data(
        sensor, ground_truth.geo_type, ground_truth.geo_value,  start_date, end_date
    )
    # TESTING IMPLEMENTATION, UNCOMMENT THIS
    # output = LocationSeries(
    #     dates=[],
    #     values=[],
    #     geo_value=ground_truth.geo_value,
    #     geo_type=ground_truth.geo_type
    # )
    # missing_dates = [int(i.strftime("%Y%m%d")) for i in date_range(str(start_date), str(end_date))]
    # #########################################################

    if not compute_missing or not missing_dates:
        return output
    # gets all available data for now, could be optimized
    response = Epidata.covidcast(data_source=sensor.source,
                                 signals=sensor.signal,
                                 time_type="day",
                                 time_values=Epidata.range(
                                     20200101,
                                     int(date.today().strftime("%Y%m%d"))),
                                 geo_value=ground_truth.geo_value,
                                 geo_type=ground_truth.geo_type)
    if response["result"] != 1:
        raise Exception(f"Bad result from Epidata: {response['message']}")
    indicator_values = LocationSeries(
        geo_value=ground_truth.geo_value,
        geo_type=ground_truth.geo_type,
        dates=[i["time_value"] for i in response["epidata"] if not np.isnan(i["value"])],
        values=[i["value"] for i in response["epidata"] if not np.isnan(i["value"])]
    )
    for day in missing_dates:
        sensor_value = compute_regression_sensor(day, indicator_values, ground_truth)
        if np.isnan(sensor_value):
            continue
        output.values.append(sensor_value)  # if np array would need to change append method
        output.dates.append(day)
        # _export_to_csv(sensor_value, sensor, ground_truth.geo_type, ground_truth.geo_value, date) # commented out for now
    return output


def _get_historical_data(indicator: SignalConfig,
                         geo_type: str,
                         geo_value: str,
                         start_date: int,
                         end_date: int) -> Tuple[LocationSeries, list]:
    """
    Query Epidata API for historical sensorization data.

    Will only return values if they are not null. If they are null or are not available, they will
    be listed as missing.

    Parameters
    ----------
    indicator
        SignalConfig specifying which sensor to retrieve.
    geo_type
        Geo type to retrieve.
    geo_value
        Geo value to retrieve.
    start_date
        First day to retrieve (inclusive).
    end_date
        Last day to retrieve (inclusive).

    Returns
    -------
        Tuple of (LocationSeries containing non-na data, list of dates without valid data)
    """
    ########################################################################################
    # REPLACE THIS WITH Epidata.covidcast_nowcast ONCE IT IS AVAILABLE (PUBLISHED TO PYPI) #
    ########################################################################################
    response = Epidata.covidcast(data_source=indicator.source,
                                         signals=indicator.signal,
                                         time_type="day",
                                         geo_type=geo_type,
                                         time_values=Epidata.range(start_date, end_date),
                                         geo_value=geo_value)
                                         # sensor_name=indicator.model) not added to DB yet.
    if response["result"] == 1:
        output = LocationSeries(
            dates=[i["time_value"] for i in response["epidata"] if not np.isnan(i["value"])],
            values=[i["value"] for i in response["epidata"] if not np.isnan(i["value"])],
            geo_value=geo_value,
            geo_type=geo_type
        )
    elif response["result"] == -2:  # no results
        print("no historical results found")
        output = LocationSeries(
            dates=[],
            values=[],
            geo_value=geo_value,
            geo_type=geo_type
        )
    else:
        raise Exception(f"Bad result from Epidata: {response['message']}")

    all_dates = [int(i.strftime("%Y%m%d")) for i in date_range(str(start_date), str(end_date))]
    missing_dates = [i for i in all_dates if i not in output.dates]
    return output, missing_dates


def _export_to_csv(value,
                   sensor,
                   geo_type,
                   geo_value,
                   date,
                   receiving_dir="/common/covidcast_nowcast/receiving" # convert this to params file
                   ) -> str:
    """Save value to csv for upload to epidata database"""
    export_dir = os.path.join(receiving_dir, sensor.signal)
    os.makedirs(export_dir, exist_ok=True)
    export_file = os.path.join(export_dir, f"{date}_{geo_type}_{sensor.signal}.csv")
    with open(export_file, "w") as f:
        f.write("sensor,geo_value,value\n")
        f.write(f"{sensor.name},{geo_value},{value}\n")
    return export_file

