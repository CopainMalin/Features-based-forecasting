from numpy import (
    array,
    ndarray,
    median,
    mean,
    nanmean,
    var,
    diff,
    log2,
)
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy.signal import welch
from hurst import compute_Hc


def __compute_STL(serie: ndarray, period: int) -> DecomposeResult:
    """Private method used to compute the STL using LOESS decomposition.

    Args:
        serie (ndarray): The time series.
        period (int): The seasonal period.

    Returns:
        DecomposeResult: The fitted STL object.
    """
    return STL(serie, period=period).fit()


def seasonal_strength(serie: ndarray, period: int) -> float:
    """Measure the strength of the seasonal component.

    Args:
        serie (ndarray): The time series.
        period (int): The seasonal period.

    Returns:
        float: The computed coefficient.
    """
    decomposition = __compute_STL(serie, period)
    return max(
        0,
        (
            1
            - decomposition.resid.var()
            / (decomposition.resid + decomposition.seasonal).var()
        ),
    )


def trend_strength(serie: ndarray, period: int) -> float:
    """Measure the strength of the trend component.

    Args:
        serie (ndarray): The time series.
        period (int): The seasonal period.

    Returns:
        float: The computed coefficient.
    """
    decomposition = __compute_STL(serie, period)
    return max(
        0,
        (
            1
            - decomposition.resid.var()
            / (decomposition.resid + decomposition.trend).var()
        ),
    )


def spikiness(ndarray: array) -> float:
    """
    "Hyndman defines the spikiness of a time series as a measure of the proportion of non-zero values
    that are larger than the median. It is essentially a way to measure the concentration of high values in a time series."

    Args:
        serie (ndarray): The time series.

    Returns:
        float: The computed coefficient.
    """
    return sum(ndarray > median(ndarray)) / ndarray.shape[0]


def lumpiness(serie: ndarray) -> float:
    """
    "Lumpiness is a measure of the variability of the series.
    Hyndman defines lumpiness as the ratio of the variance to the square of the mean.
    Higher values of lumpiness indicate greater variability relative to the mean."

    Args:
        serie (ndarray): The time series.

    Returns:
        float: The computed coefficient.
    """
    return var(serie) / mean(serie) ** 2


def curvature(serie: ndarray) -> float:
    """
    "Curvature measures the degree to which the time series exhibits a curvilinear pattern.
    Hyndman often uses the second order difference of the time series to assess its curvature.
    A positive second order difference suggests a convex pattern,
    while a negative second order difference indicates a concave pattern"

    Args:
        serie (ndarray): The time series.

    Returns:
        float: The computed coefficient.
    """
    return nanmean(diff(diff(serie)))


def spectral_entropy(serie: ndarray) -> float:
    """
    A way to determine if a signal is forecastable, i.e there is a strong signal/noise ratio.
    low entropy -> very few noise.
    high entropy -> high noise.

    Args:
        serie (ndarray): The time series.

    Returns:
        float: The computed coefficient.
    """
    _, power_density = welch(serie)
    normalized_power_density = power_density / sum(power_density)
    return -sum(normalized_power_density * log2(normalized_power_density))


def hurst_exponent(serie: ndarray) -> float:
    """
    A coefficient that indicates the nature of the time series, i.e if there is some mean reversion, persistence or if the
    time series can be described as a Wiener process.

    Args:
        serie (ndarray): The time series.

    Returns:
        float: The computed coefficient.
    """
    return compute_Hc(serie, simplified=True)[0]


def adf_pvalue(serie: ndarray) -> float:
    """Return the p-value of the augmented Dickey Fuller test.
    H0 : No unit root (stationarity).
    H1 : Unit root (non stationarity, explosive datas).

    Args:
        serie (ndarray): The time serie to compute the adf test on.

    Returns:
        float: The p value of the adf test.
    """
    stat, p_value, _, _, _, _ = adfuller(serie)
    return p_value


def autocorrelation(serie: ndarray, seasonal_period=int) -> [float]:
    """
    Coefficients measuring the correlation between the value of the serie a time t, and values at time t-1, t-2, ...

    Args:
        serie (ndarray): The time series.

    Returns:
        [float]: The computed autocorolletation, each element corresponding to its associated lag.
                For instance, value at index 1 = corr(y_t, y_(t-1)).
    """
    return acf(x=serie, nlags=seasonal_period + 5)


def partial_autocorrelation(serie: ndarray, seasonal_period=int) -> [float]:
    """
    Coefficients measuring the partial correlation between the value of the serie a time t, and values at time t-1, t-2, ...
    The partial autocorrelation is the autocorrelation corrected of the impact of the previous lags. We can see it as the
    autocorrelation without the redundancy.

    Args:
        serie (ndarray): The time series.

    Returns:
        [float]: The computed autocorolletation, each element corresponding to its associated lag.
                For instance, value at index 1 = corr(y_t, y_(t-1)).
    """
    return pacf(x=serie, nlags=seasonal_period + 5)
