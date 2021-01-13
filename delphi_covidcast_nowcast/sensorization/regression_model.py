import numpy as np

from ..data_containers import LocationSeries


def compute_regression_sensor(date: int,
                              covariate: LocationSeries,
                              response: LocationSeries,
                              include_intercept=False,
                              lambda_: float = 0.1) -> float:
    """
    Fit regression model and get sensorization value for a given date.

    This takes two LocationSeries objects for a covariate and response as well as a date to
    predict and some model parameters. The model is trained on all data before the specified date,
    and then the predictor at the given date is fed into the model to get the returned sensor value
    for that day.

    For now, this function assumes there are no gaps in the data.

    It does not normalize the data yet.

    Parameters
    ----------
    date
        date to get sensor value for
    covariate
        LocationSeries containing covariate values.
    response
        LocationSeries containing response values.
    include_intercept
        Boolean on whether or not to include intercept.
    lambda_
        l2 regularization coefficient

    Returns
    -------
        Float value of sensor on `date`
    """
    # fill in gaps in data if any missing dates (e.g. polynomial imputation)?
    idx = covariate.dates.index(date)
    train_Y = response.values[:idx]
    train_covariates = covariate.values[:idx]
    # error if not enough data in either values or response

    date_X = np.hstack((1, covariate.values[idx])) if include_intercept else covariate.values[idx]
    X = np.ones((len(train_covariates), 1 + include_intercept))
    X[:, -1] = train_covariates
    B = np.linalg.inv(X.T @ X + lambda_ * np.eye(1 + include_intercept)) @ X.T @ train_Y
    return date_X @ B
