from ..data_containers import LocationSeries


def compute_regression_sensor(date: int,
                              values: LocationSeries,
                              response: LocationSeries,
                              other_params=None) -> float:
    """Fit model and get sensorization value for a given date."""
    # select dates required for training and predicting a value for "date" e.g. last 60 days
    # error if not enough data in either values or response
    # for specified date, compute the sensorization value and return it
    pass
