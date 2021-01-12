"""Retrieve reporting delay distribution"""

from typing import List

import datetime
import pandas as pd
import scipy.stats as stats


def get_florida_delay_distribution(update: bool = False) -> List[float]:
    if update:
        florida_linelist = pd.read_csv(
            "https://www.arcgis.com/sharing/rest/content/items/4cc62b3a510949c7a8167f6baa3e069d/data",
            parse_dates=["Case_", "EventDate", "ChartDate"])

        delay_df = florida_linelist[
            florida_linelist.EventDate > datetime.datetime(2020, 5, 1)]
        delay_df["delay"] = (delay_df.ChartDate - delay_df.EventDate).dt.days
        delay_df = delay_df[delay_df.delay.gt(0) & delay_df.delay.lt(45)]
        coefs = stats.gamma.fit(delay_df.delay, floc=0)
    else:
        coefs = (1.4948103204081697, 0, 4.282169385049879)

    # discretized distribution
    delay_gam = stats.gamma(*coefs)
    delay_dist = delay_gam.pdf(range(1, 45))
    delay_dist /= delay_dist.sum()
    return list(delay_dist)
