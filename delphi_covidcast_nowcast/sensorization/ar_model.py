import numpy as np

from ..data_containers import LocationSeries


def compute_ar_sensor(date: int,
                      values: LocationSeries,
                      ar_size: int = 2,
                      include_intercept: bool = False,
                      lambda_: float = 0.1) -> float:
    """
    Fit AR model and get sensorization value for a given date.

    This takes in a LocationSeries objects for the quantity of interest as well as a date to
    predict and some model parameters. The model is trained on all data before the specified date,
    and then the predictor at the given date is fed into the model to get the returned sensor value
    for that day.

    For now, this function assumes there are no gaps in the data.

    It does not normalize the data yet.

    Parameters
    ----------
    date
        date to get sensor value for
    values
        LocationSeries containing covariate values.
    ar_size
        Order of autoregressive model.
    include_intercept
        Boolean on whether or not to include intercept.
    lambda_
        l2 regularization coefficient

    Returns
    -------
        Float value of sensor on `date`
    """
    window = values.get_data_range(min(values.dates), date)
    Yhat = ar_predict(len(window)-1, np.array(values.values), ar_size, include_intercept, lambda_)
    if Yhat is None:
        return np.nan
    # should we set a seed here?
    # np.random.seed(date) maybe?

    # ground truth in some locations is a zero vector, which leads to perfect
    # AR fit, zero variance, and a singular covariance matrix so as a small
    # hack, add some small noise.
    Yhat += np.random.normal(0, 0.1)

    # as a huge hack, add more noise to prevent AR from unreasonably dominating
    # the nowcast since AR3 can nearly exactly predict some trendfiltered
    # curves.
    Yhat += np.random.normal(0, 0.1 * np.maximum(0, np.mean(Yhat)))
    return Yhat


def ar_predict(idx, values, ar_size, include_intercept, lambda_):
    # taken from https://github.com/dfarrow0/covidcast-nowcast/tree/dfarrow/sf/src/sf
    #
    # predict the value at values[idx] using values[idx - ar_size:idx]
    # to do that, train on all values[:idx]
    # note that an L2 penalty is applied since sometimes there is colinearity,
    # like when `values` is all zeros.
    # TODO: L2 is implemented incorrectly. ideally covariates would be
    # normalized before adding the penalty (so as not to unfairly penalize
    # covariates with high variance), but here they're not being normalized.
    # probably doesn't matter too much for now, but something to fix later.
    num_covariates = ar_size
    if include_intercept:
        num_covariates += 1
    num_observations = idx - ar_size
    if num_observations < 2 * num_covariates:
        # require some minimum number of samples
        return None

    # fairly standard OLS, maybe with intercept, and with L2 penalty
    X = np.zeros((num_observations, num_covariates))
    if include_intercept:
        X[:, -1] = 1
    for j in range(ar_size):
        X[:, j] = values[j:idx - ar_size + j]
    Y = values[ar_size:idx, None]
    X = np.vstack((X, lambda_ * np.eye(num_covariates)))
    Y = np.vstack((Y, np.zeros((num_covariates, 1))))
    B = np.linalg.inv(X.T @ X) @ X.T @ Y

    # given the model fit above, predict the value at `idx`
    x = values[None, idx - ar_size:idx]
    if include_intercept:
        x = np.hstack((x, [[1]]))

    # return model and estimate at `idx`
    return (x @ B)[0, 0]

