"""Functions to run sensorization."""

from typing import List, Tuple

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
        TBDL
    """
    # api_sensors = []
    # ar_sensors = []
    # for each location:
    #     ar_sensors.append(get_ar_sensor(location, values, ...))
    #     for each indicator:
    #         sensors.append(get_indicator_sensor(location, indicator, input_dates)
    # return api_sensors + ar_sensors
    pass


def get_ar_sensor():
    # get existing sensor values from DB
    # for dates that arent available (hopefully just today), grab necessary window of indicator values
    # return sensor values up to and including today (current sensor value)
    pass


def get_indicator_sensor():
    # same as ar_sensor but with different signals
    pass