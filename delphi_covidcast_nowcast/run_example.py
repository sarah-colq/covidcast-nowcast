"""Small example."""

import delphi_covidcast_nowcast.deconvolution.delay_kernel as delay
import delphi_covidcast_nowcast.nowcast as nowcast


def main():
    input_dates = list(range(20200601, 20200610))
    input_locations = [('pa', 'state'),
                       ('42003', 'county')]
    sensor_indicators = [('auto-reg', 'ar3'),
                         ('fb-survey', 'smoothed_hh_cmnty_cli'),
                         ('doctor-visits', 'smoothed_adj_cli')]
    convolved_truth_indicator = ('usa-facts', 'confirmed_incidence_num')
    kernel = delay.get_florida_delay_distribution()
    nowcast_dates = [20200610]

    infections = nowcast.nowcast(input_dates, input_locations,
                                 sensor_indicators, convolved_truth_indicator,
                                 kernel, nowcast_dates)


if __name__ == '__main__':
    main()
