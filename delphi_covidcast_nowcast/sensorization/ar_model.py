import numpy as np

from ..data_containers import LocationSeries


def compute_ar_sensor(date: int,
                      values: LocationSeries,
                      ar_size=2,
                      include_intercept=False,
                      lambda_=0.1)-> float:
    """Fit model and get sensorization value for a given date."""
    # fill in gaps in data if any missing dates (e.g. polynomial imputation)?
    idx = values.dates.index(date)
    Yhat = ar_predict(idx, np.array(values.values), ar_size, include_intercept, lambda_)[1]

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
    return B, (x @ B)[0, 0]

