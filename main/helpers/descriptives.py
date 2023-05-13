import statistics as stats
from math import *  # libraries
from statistics import *

import numpy
import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable

###Setting Functions
def descriptives(data):
    Data = numpy.array(data)
    d_mean = round(mean(Data), 2)
    d_trim = round(stats.trim_mean(Data, 0.1), 2)
    d_median = round(median(Data), 2)
    d_var = round(sum((Data - d_mean) ** 2) / len(Data - 1), 2)
    d_se = round(sqrt(d_var), 2)
    q3, q1 = np.percentile(Data, [75, 25])
    d_iqr = round(q3 - q1, 2)
    d_range = [round(min(Data), 2), round(max(Data), 2)]
    d_deviance = Data - d_mean
    d_MAD = round(sum(abs(Data - d_mean)) / len(Data), 2)
    d_skewness = round(
        sum(numpy.power(d_deviance, 3)) / ((len(Data) - 1) * pow(d_se, 3)), 2
    )
    d_kurtosis = round(sum(numpy.power(d_deviance, 4)) / (len(Data) * pow(d_se, 4)), 2)
    vec_df = {
        "Descriptive": [
            "Mean",
            "Trimmed 10%",
            "Median",
            "Variance",
            "Standard Dev",
            "IQR",
            "Range",
            "MAD",
            "Skewness",
            "Kurtosis",
        ],
        "Result": [
            d_mean,
            d_trim,
            d_median,
            d_var,
            d_se,
            d_iqr,
            d_range,
            d_MAD,
            d_skewness,
            d_kurtosis,
        ],
    }
    df = pd.DataFrame(vec_df)
    return df