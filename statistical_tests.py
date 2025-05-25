import numpy as np
import scipy.stats as stats


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
    references:
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
