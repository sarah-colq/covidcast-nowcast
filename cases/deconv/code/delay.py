"""
Estimate symptom-onset to reporting delay distribution.

Created: 2020-09-23
"""

# standard
import tarfile

# third party
import pandas as pd
import requests
from scipy.stats import gamma


def get_international_delays(data_path=None, download=False):
    if data_path is None:
        if not download:
            print("needs data_path or download")
            return False
        data_path = "./linelist.tar.gz"  # store in current working dir

    if download:
        url = "https://github.com/beoutbreakprepared/nCoV2019/raw/master/latest_data/latestdata.tar.gz"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(data_path, 'wb') as f:
                f.write(response.content)

    with tarfile.open(data_path, "r:*") as tar:
        df = pd.read_csv(tar.extractfile(tar.getnames()[0]),
                         usecols=["country",
                                  "date_onset_symptoms",
                                  "date_confirmation"],
                         low_memory=False)

    df.dropna(inplace=True)

    # handle dates - follow same cleaning process as rt.live
    df = df[df['date_confirmation'].str.len().eq(10) &
            df['date_onset_symptoms'].str.len().eq(10)]  # proper date format

    # bad dates that were reversed
    df = df.replace("01.31.2020", "31.01.2020")
    df = df.replace("31.04.2020", "01.05.2020")

    # convert to dates
    df['date_onset_symptoms'] = pd.to_datetime(df['date_onset_symptoms'],
                                               format='%d.%m.%Y')
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'],
                                             format='%d.%m.%Y')

    # drop Mexico due to lots of cases confirmed on same date, no
    # association with symptom onset (taken from rt.live)
    df = df[df.country.ne("Mexico")]
    delays = (df.date_confirmation - df.date_onset_symptoms).dt.days

    # drop delays longer than 30 days (approx 4 weeks)
    delays = delays[delays.le(60) & delays.gt(0)].reset_index(drop=True)
    return delays


def get_delay_distribution(delays):
    """
    Fit a gamma delay distribution.

    Args:
        delays: array of delay times

    Returns:
        array of delay probabilities
    """
    gam = gamma(*gamma.fit(delays, floc=0))

    # discretized distribution
    delay_pr = gam.pdf(range(1, 60))
    delay_pr /= delay_pr.sum()
    return delay_pr
