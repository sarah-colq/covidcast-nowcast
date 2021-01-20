from unittest.mock import patch

import numpy as np

from delphi_covidcast_nowcast.sensorization.ar_model import compute_ar_sensor
from delphi_covidcast_nowcast.data_containers import LocationSeries


class TestComputeRegressionSensor:

    @patch("numpy.random.normal")
    def test_compute_regression_sensor_intercept(self, random_normal):
        """Verified with ar.ols(x, FALSE, ar_size, intercept=TRUE, demean=FALSE)."""
        random_normal.return_value = 0
        values = LocationSeries(
            values=[-4.27815483, -4.83962077, -4.09548122, -3.86647783, -2.64494168, -3.99573135,
                    -3.48248410, -2.77490127, -3.64162355, -2.57628910, -2.46793048, -3.20454941,
                    -1.77057154, -0.02058535, 0.81182691, 0.32741982],
            dates=[20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108,
                   20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116])
        assert np.isclose(
            compute_ar_sensor(20200115, values, 1, True, 0, False),
            -0.09105891 + 0.87530957 * -0.02058535
        )
        assert np.isclose(
            compute_ar_sensor(20200115, values, 2, True, 0, False),
            0.31865395 + 0.64751725 * -0.02058535 + 0.30760218 * -1.77057154
        )

    @patch("numpy.random.normal")
    def test_compute_regression_sensor_no_intercept(self, random_normal):
        """Verified with ar.ols(x, FALSE, ar_size, intercept=False, demean=FALSE)."""
        random_normal.return_value = 0
        values = LocationSeries(
            values=[-4.27815483, -4.83962077, -4.09548122, -3.86647783, -2.64494168, -3.99573135,
                    -3.48248410, -2.77490127, -3.64162355, -2.57628910, -2.46793048, -3.20454941,
                    -1.77057154, -0.02058535, 0.81182691, 0.32741982],
            dates=[20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108,
                   20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116])
        assert np.isclose(
            compute_ar_sensor(20200115, values, 1, False, 0, False),
            0.90084256 * -0.02058535
        )
        assert np.isclose(
            compute_ar_sensor(20200115, values, 2, False, 0, False),
            0.61458575 * -0.02058535 + 0.25038242 * -1.77057154
        )

    def test_compute_regression_sensor_seed(self):
        """Test same result over 50 runs"""
        values = LocationSeries(
            values=[-4.27815483, -4.83962077, -4.09548122, -3.86647783, -2.64494168, -3.99573135,
                    -3.48248410, -2.77490127, -3.64162355, -2.57628910, -2.46793048, -3.20454941,
                    -1.77057154, -0.02058535, 0.81182691, 0.32741982],
            dates=[20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108,
                   20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116])
        assert len(set(compute_ar_sensor(20200115, values, 1, False, 0) for i in range(50))) == 1

    def test_compute_regression_sensor_insufficient_data(self):
        values = LocationSeries(
            values=[-4.27815483, -4.83962077],
            dates=[20200101, 20200102])
        assert np.isnan(compute_ar_sensor(20200102, values))

    @patch("numpy.random.normal")
    def test_compute_regression_sensor_standardize_intercept(self, random_normal):
        """Test standardizing sets intercept to false."""
        random_normal.return_value = 0
        values = LocationSeries(
            values=[-4.27815483, -4.83962077, -4.09548122, -3.86647783, -2.64494168, -3.99573135,
                    -3.48248410, -2.77490127, -3.64162355, -2.57628910, -2.46793048, -3.20454941,
                    -1.77057154, -0.02058535, 0.81182691, 0.32741982],
            dates=[20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108,
                   20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116])
        assert np.isclose(
            compute_ar_sensor(20200115, values, 1, True, 0, True),
            compute_ar_sensor(20200115, values, 1, False, 0, True),
        )

    def test_compute_regression_sensor_standardize(self):
        # TODO VALIDATE OUTPUT. Currently just runs them to make sure they don't break.
        # the formulas we use look sound, but would be nice to have another package validate.
        values = LocationSeries(
            values=[-4.27815483, -4.83962077, -4.09548122, -3.86647783, -2.64494168, -3.99573135,
                    -3.48248410, -2.77490127, -3.64162355, -2.57628910, -2.46793048, -3.20454941,
                    -1.77057154, -0.02058535, 0.81182691, 0.32741982],
            dates=[20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108,
                   20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116])
        compute_ar_sensor(20200115, values, 1, False, 0, True),
        assert 1

    def test_compute_regression_sensor_regularize(self):
        # TODO VALIDATE OUTPUT. Currently just runs them to make sure they don't break.
        # the formulas we use look sound, but would be nice to have another package validate.
        values = LocationSeries(
            values=[-4.27815483, -4.83962077, -4.09548122, -3.86647783, -2.64494168, -3.99573135,
                    -3.48248410, -2.77490127, -3.64162355, -2.57628910, -2.46793048, -3.20454941,
                    -1.77057154, -0.02058535, 0.81182691, 0.32741982],
            dates=[20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108,
                   20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116])
        compute_ar_sensor(20200115, values, 1, False, 1, True),
        assert 1
