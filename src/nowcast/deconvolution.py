"""Generate ground truth signal."""

from typing import List, Tuple

import numpy as np

def deconv_signal(ground_truth_indicator: Tuple[str, str],
                  input_dates: List[int],
                  input_locations: List[Tuple[str, str]],
                  kernel: np.array
                  ) -> np.ndarray:
    """
    Compute ground truth signal value by deconvolving an indicator with a delay distribution.

    A trend filtering penalty is applied to smooth the output.

    Parameters
    ----------
    ground_truth_indicator
        Tuple specifying the source and signal of interest to nowcast.
    input_dates
        List of dates to train data on and get nowcasts for.
    input_locations
        List of (location, geo_type) tuples specifying locations to train and obtain nowcasts for.
    kernel
        Delay distribution from infection to report.

    Returns
    -------
        Matrix of ground truth values for each location.
    """
    # num_locs = len(input_locations)
    # num_dates = len(input_dates)
    # deconv_infections = np.full((num_locs, num_dates), np.nan)
    # for each input_location i:
    #     raw_cases = Epidata.covidcast(source, signal...)
    #     deconv_infections[i, :] = DeconvADMM.fit_cv(raw_cases, kernel, ...)
    # return deconv_infections
    pass

def fit(signal, # TODO
        kernel: np.array,
        k: int,
        max_iters: int
        ):  # TODO return type
    # ADMM code (already completed)
    pass


def fit_cv(signal,  # TODO np.ndarray?
           kernel: np.array,
           k: int,
           max_iters: int,
           cv_grid  # TODO list?
           ):  # TODO return type
    # for param in cv_grid:
    #     est = DeconvADMM.fit(signal[train_inds], kernel)
    #     loss = mae(predict(est), signal[test_inds])
    # pass