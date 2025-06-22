import numpy as np
import pandas as pd
from typing import List
import scipy.stats as stats
from IPython.display import display
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor


def pttest(y: list[int], yhat: list[int]):
    
    '''
    The Pesaran-Timmermann test is based on the proportion
    of times that the direction of change in y is correctly
    predicted in the sample. It requires that the probability
    of changes in the direction of y and yhat is time-invariant
    and does not take extreme values of 0 and 1.
    
    y: variable of interest
    yhat: predictor of y
    returns: Directional Accuracy Score, Pesaran-Timmermann statistic and its p-value
    
    References:
    https://gist.github.com/vpekar/df58ac8f07ec9d4ef24bcf1c176812b0
    https://www.jstor.org/stable/1391822

    Example usage:
    a = np.array([23, -2, 56, 51, 4, -45, -12, -24, -51, 78, -6, -7, -39, 31, 35])
    b = np.array([14, 3, 45, 23, -5, -56, 4, -11, -34, 29, 3, -11, -12, 24, 3])
    dac, pt, pval = pttest(a, b)
    print(f"Directional Accuracy: {round(dac, 2)}, PT stat: {round(pt, 2)}, p-value: {round(pval, 2)}")
    '''
    
    size = y.shape[0]
    pyz = np.sum(np.sign(y) == np.sign(yhat)) / size
    py = np.sum(y > 0) / size
    qy = py * (1 - py) / size
    pz = np.sum(yhat > 0) / size
    qz = pz * (1 - pz) / size
    p = py * pz + (1 - py) * (1 - pz)
    v = p*(1 - p) / size
    w = ((2*py - 1)**2) * qz + ((2*pz - 1)**2) * qy + 4*qy*qz
    pt = (pyz - p) / (np.sqrt(v - w))
    pval = 1 - stats.norm.cdf(pt, 0, 1)
    
    return pyz, pt, pval

def print_vif(
    df: pd.DataFrame = None,
    exog: List[str] = None,
) -> None:
    """Calculate the VIF for each variable using statsmodels
    As mentioned by Josef Perktold, the function's author,
    variance_inflation_factor expects the presence of
    a constant in the matrix of explanatory variables.
    
    Reference:
    https://github.com/statsmodels/statsmodels/issues/2376

    :param df: dataframe, defaults to None
    :type df: pd.DataFrame, optional
    :param exog: list with exogenous variables names, defaults to None
    :type exog: List, optional

    Example usage:
    np.random.seed(0)
    df = pd.DataFrame({
        'const': 1,
        'x1': np.random.normal(size=100),
        'x2': np.random.normal(size=100),
        'x3': np.random.normal(size=100)
    })
    print_vif(df=df, exog=['const', 'x1', 'x2', 'x3'])
    """
    
    vif = dict()
    for i, var in enumerate(exog):
        vif[var] = variance_inflation_factor(exog=df[exog], exog_idx=i).round(2)

    display(pd.DataFrame.from_dict(data=vif, orient='index', columns=['VIF']).sort_values(by=['VIF'], ascending=False))
    
def perform_adf_test(series: pd.Series = None, regression: str = 'c') -> float:
    """
    Perform Augmented Dickey-Fuller test to check for stationarity in a time series.

    :param series: Time series data, defaults to None
    :type series: pd.Series, optional
    :param regression: Type of regression ('c', 'ct', 'ctt', 'n'), defaults to 'c'
    :type regression: str, optional
    :return: p-value of the test
    :rtype: float

    Stationarity means that the statistical properties of a time series, i.e. mean, variance and covariance do not change over time. 
    Many statistical models require the series to be stationary to make effective and precise predictions.
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root (series is not stationary).
    The alternative hypothesis is that there is no unit root.

    Example usage:
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.stattools import adfuller

    # Generate a random walk (non-stationary)
    np.random.seed(0)
    x = np.random.normal(size=100).cumsum()
    series = pd.Series(x)

    pval = perform_adf_test(series)
    print(f"ADF test p-value: {pval}")
    """
    pval = np.round(adfuller(x=series, regression=regression, autolag=None)[1], 3)
    return pval

