"""
Estimate infection curve

Created: 2020-09-09
Last modified: 2020-09-23
"""
# third party
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# r imports
genlasso = importr('genlasso')
rlist2dict = lambda x: dict(x.items())
rfloat2arr = lambda x: np.array(x)

# first party
from .conv1d import Conv1D


class InfectionCurve:

    def __init__(self, delay, verbose=False):
        """

        Args:
            delay: 1D array of delay distribution probabilities
            verbose: bool if output should be printed
        """
        self.delay = delay
        # TF via genlasso package
        robjects.r(
            '''
            tf_predict <- function(tf_fit, x.new=NULL, lambda=NULL) {
                predict(tf_fit, x.new=x.new, lambda=lambda)
            }

            tf_predict_cv_ <- function(tf_fit, n_folds=5) {
                cv = genlasso::cv.trendfilter(tf_fit, k=n_folds)
                min.lam = cv$lambda.min
                print(paste("Min lambda:", min.lam))
                preds = predict(tf_fit, lambda=min.lam)
                list(lam=min.lam, preds=preds$fit)
            }

            tf_predict_cv <- function(tf_fit, n_folds=5) {
                utils::capture.output(res <- tf_predict_cv_(tf_fit, n_folds=n_folds))
                res
            }

           '''
        )
        if verbose:
            self.tf_predict = robjects.r['tf_predict_cv_']
        else:
            self.tf_predict = robjects.r['tf_predict_cv']

    def get_infection_curve(self, y, k=2, n_folds=3):
        """
        Estimate infections via ADMM TF framework:

        x_tilde = argmin ||W^{-1}y - x||_2^2 + lam*||Dx||_1
                    x

        where W is the convolution matrix, and D is the discrete
        difference operator of order k+1.
        """
        n = y.shape[0]
        W = Conv1D.get_conv_matrix(y, self.delay)[:n, ]
        r_y = robjects.FloatVector(np.linalg.inv(W) @ y)
        mod = genlasso.trendfilter(r_y, ord=k)
        r_pred = rlist2dict(self.tf_predict(mod, n_folds))['preds']

        return rfloat2arr(r_pred).flatten()
