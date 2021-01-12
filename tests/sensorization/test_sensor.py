from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from delphi_covidcast_nowcast.sensorization.sensor import \
    get_sensors, get_sensor_values, _get_historical_data
from delphi_covidcast_nowcast.data_containers import LocationSeries, SignalConfig


class TestGetSensors:

    @patch("delphi_covidcast_nowcast.sensorization.sensor.get_sensor_values")
    def test_get_sensors(self, get_sensor_values):
        """Test sensors are obtained correctly."""
        get_sensor_values.side_effect = [LocationSeries("w"),
                                         LocationSeries("x"),
                                         LocationSeries("y"),
                                         LocationSeries("z")]
        test_sensors = [SignalConfig("1"), SignalConfig("2")]
        test_ground_truths = [LocationSeries(geo_value="ca", geo_type="state", values=[np.nan, 1]),
                              LocationSeries(geo_value="pa", geo_type="state", values=[2, 3]),
                              LocationSeries(geo_value="ak", geo_type="state", values=[4, 5])]
        assert get_sensors(None, None, test_sensors, test_ground_truths, True) == {
            SignalConfig("1"): [LocationSeries("w"), LocationSeries("y")],
            SignalConfig("2"): [LocationSeries("x"), LocationSeries("z")]
        }


class TestGetSensorValues:

    @patch("delphi_covidcast_nowcast.sensorization.sensor._get_historical_data")
    def test_get_sensor_values_no_missing(self, historical):
        """Test output is just returned if no missing dates"""
        historical.return_value = ("output", [])
        test_ground_truth = LocationSeries(geo_value="ca", geo_type="state")
        assert get_sensor_values(None, None, None, test_ground_truth, True) == "output"

    @patch("delphi_covidcast_nowcast.sensorization.sensor._get_historical_data")
    def test_get_sensor_values_no_compute(self, historical):
        """Test output is just returned in compute_missing=False"""
        historical.return_value = ("output", [20200101])
        test_ground_truth = LocationSeries(geo_value="ca", geo_type="state")
        assert get_sensor_values(None, None, None, test_ground_truth, False) == "output"


    @patch("covidcast.signal")
    @patch("delphi_covidcast_nowcast.sensorization.sensor.compute_ar_sensor")
    @patch("delphi_covidcast_nowcast.sensorization.sensor.compute_regression_sensor")
    @patch("delphi_covidcast_nowcast.sensorization.sensor._get_historical_data")
    @patch("delphi_covidcast_nowcast.sensorization.sensor._export_to_csv")
    def test_get_sensor_values_compute(self, export_csv, historical, regression_sensor, ar_sensor, covidcast_signal):
        """Test computation functions are called for missing dates"""
        export_csv.return_value = None
        historical.return_value = (LocationSeries(values=[], dates=[]), [20200101])
        regression_sensor.return_value = "regression value"
        ar_sensor.return_value = "ar value"
        covidcast_signal.return_value = MagicMock(time_value=None, values=None)
        test_ground_truth = LocationSeries(geo_value="ca", geo_type="state")
        # test invalid sensor
        invalid_sensor = SignalConfig("1", model="not_valid")
        with pytest.raises(ValueError):
            get_sensor_values(invalid_sensor, None, None, test_ground_truth, True)

        # instantiate new object for return value and test ar sensor
        historical.return_value = (LocationSeries(values=[], dates=[]), [20200101])
        ar_sensor = SignalConfig("1", model="ar")
        assert get_sensor_values(ar_sensor, None, None, test_ground_truth, True) == \
               LocationSeries(values=["ar value"], dates=[20200101])

        # instantiate new object for return value and test regression sensor
        historical.return_value = (LocationSeries(values=[], dates=[]), [20200101])
        regression_sensor = SignalConfig("1", model="regression")
        assert get_sensor_values(regression_sensor, None, None, test_ground_truth, True) == \
               LocationSeries(values=["regression value"], dates=[20200101])


class TestGetHistorcalData:

    @patch("delphi_covidcast_nowcast.sensorization.sensor.Epidata.covidcast")
    def test__get_historical_data(self, epidata):
        test_sensor = SignalConfig()
        epidata.return_value = {
              'result': 1,
              'epidata': [{
                'time_value': 20200101,
                'value': 3.5,
              }, {
                'time_value': 20200103,
                'value': 2.5,
              }],
              'message': 'success',
            }

        _get_historical_data(test_sensor, "geotype", "geoval", 20200101, 20200104)


class TestExportToCSV:

    def test__export_to_csv(self):
        pass