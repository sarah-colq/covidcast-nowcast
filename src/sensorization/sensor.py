"""Functions to run sensorization."""

from typing import List, Tuple, Dict

import numpy as np
import covidcast

from ..data_containers import LocationSeries
from .model import compute_ar_sensor, compute_regression_sensor


def get_sensors(start_date: int,
				end_date: int,
				locations: List[Tuple[str, str]],
				ground_truth: List[LocationSeries],
				sensor_indicators: List[Tuple[str, str, str]],
				ar_indicators: List[Tuple[str, str, str]],
				compute_missing: bool,
				) -> Dict[Tuple[str, str, str], LocationSeries]:
	"""
	Return sensorization from start to end date at given locations for specified sensors.

	Only locations that have complete ground truths (no nans) will have sensorization values computed.

	Parameters
	----------
	start_date
	end_date
	locations
	ground_truth
	sensor_indicators
	ar_indicators
	compute_missing

	Returns
	-------

	"""
	output = {}
	if compute_missing == False:
		# is compute_missing is false, we'll try to return data from all locations
		locations_to_get = locations
	else:
		# else we'll only get data for the locations that have no nans at all,
		# which is a bit of a rough approach but we can tune later
		avail_truth = [y for y in ground_truth if not np.all(np.isnan(y.values))]
		unavail_loc = set(x[0] for x in locations) - set(y.geo_value for y in avail_truth)
		if unavail_loc:
			print(f"unavailable locations {unavail_loc}")
		locations_to_get = avail_truth

	for location in locations_to_get:
		for indicator in sensor_indicators:
			output[indicator] = get_regression_sensor(indicator, location, start_date, end_date,
													  ground_truth, compute_missing)
		for indicator in ar_indicators:
			output[indicator] = get_ar_sensor(indicator, location, start_date, end_date,
											  ground_truth, compute_missing)
	return output



def get_ar_sensor(location, ground_truth_indicator, min_date, max_date, ground_truth, compute_missing):
	"""
	Return AR sensorization values, using available historical data if specified.

	Parameters
	----------
	location
	ground_truth_indicator
	min_date
	max_date
	ground_truth
	compute_missing

	Returns
	-------

	"""
	# left out recompute_all_data argument for now just to keep things simple
	output, missing_dates = _get_historical_data(ground_truth_indicator, min_date, max_date)
	if missing_dates and compute_missing:
		# SHOULD THIS BE ALL AVAILABLE DATES IN CASE COMPUTE_AR_SENSOR WINDOW EXTENDS BEFORE MIN_DATE? I THINK SO
		indicator_values = covidcast.signal(location) # output should contain both values and dates
		for date in missing_dates:
			sensor_value = compute_ar_sensor(date, indicator_values, ground_truth, other params)
			output.append(sensor_value, date)
			# export sensor_value for upload
	return output


def get_regression_sensor(location, ground_truth_indicator, min_date, max_date, ground_truth,
				  compute_missing):
	"""
	Get regression sensorization values using available historical data if specified.

	Parameters
	----------
	location
	ground_truth_indicator
	min_date
	max_date
	ground_truth
	compute_missing

	Returns
	-------

	"""
	# left out recompute_all_data argument for now just to keep things simple
	output, missing_dates = _get_historical_data(ground_truth_indicator, min_date, max_date)
	if missing_dates and compute_missing:
		# SHOULD THIS BE ALL AVAILABLE DATES IN CASE COMPUTE_AR_SENSOR WINDOW EXTENDS BEFORE MIN_DATE? I THINK SO
		indicator_values = covidcast.signal(location)  # output should contain both values and dates
		for date in missing_dates:
			sensor_value = compute_regression_sensor(date, indicator_values, ground_truth, other_params)
			output.append(sensor_value, date)
		# export sensor_value for upload
	return output



def _get_historical_data(ground_truth_indicator, min_date, max_date)
	"""Query Epidata API for historical sensorization data."""
	# make epidata.covidcast_nowcast call
	# convert data to proper format (list of(value, date) tuples?)
	# compute missing dates between min_date and max date
	return data, missing_dates


