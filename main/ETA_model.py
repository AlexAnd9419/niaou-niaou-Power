

import statistics as stats
from math import *  # libraries
from statistics import *

import statsmodels as statsmodels
from pandas.api.types import is_categorical_dtype

from scipy import stats
from typing import Callable
import pandas as pd
from math import *  # libraries
import statistics
import numpy as np
from scipy import stats
from statistics import *
import statsmodels

###Setting Functions
from scipy.stats import levene


def descriptives(data):
    Data = data
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
        sum(np.power(d_deviance, 3)) / ((len(Data) - 1) * pow(d_se, 3)), 2
    )
    d_kurtosis = round(sum(np.power(d_deviance, 4)) / (len(Data) * pow(d_se, 4)), 2)
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
def HomoscedasticityB(y, x):
    binary_variable = x
    continuous_variable = y
    test = levene(binary_variable, continuous_variable)
    return test

pd.set_option("display.max_columns", None)  # or 1000
pd.set_option("display.max_rows", 10)  # or 1000
pd.set_option("display.max_colwidth", None)  # or 199
x=np.random.normal(200,20,100)
download_time=x
x=np.random.binomial(1,0.2,100)
stealth_use=x
x1,x2,x3,x4,x5=(['Penray']*20,['Champion']*20,['Pfizer']*20,['Oakwood']*20,['Zep']*20)
x=x1+x2+x3+x4+x5
vendor=x

x=np.random.normal(7000,1000,100)
total_files=x
x=np.random.binomial(1,0.05,100)
proxy_use=x
data=pd.DataFrame({"download_time":download_time,"stealth_use":stealth_use,"vendor":vendor,"total_files":total_files,"proxy_use":proxy_use})
data['vendor']=pd.Categorical(data['vendor'])
print(descriptives(data['download_time']))
print(descriptives(data['total_files']))
y=data['download_time']
x=data.drop(columns='download_time')
x = stats.kstest(rvs=y, method="exact",cdf='norm')
x=stats.shapiro(y)
print(x)

