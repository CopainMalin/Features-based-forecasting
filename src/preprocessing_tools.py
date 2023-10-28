from pandas import Series, DataFrame
from numpy import ndarray, nanmean, nanmedian, nanquantile, nanstd
from src.features_computation_tools import (
    seasonal_strength,
    trend_strength,
    spikiness,
    lumpiness,
    curvature,
    spectral_entropy,
    hurst_exponent,
    adf_pvalue,
)


def build_rolling_features(
    serie: Series, seasonal_period: int, lags_to_consider: int = 5
) -> DataFrame:
    if seasonal_period >= 100:  # Hurst exponent needs 100 values to works
        rolling_features = serie.rolling(window=seasonal_period).aggregate(
            {
                "mean": nanmean,
                "median": nanmedian,
                "std": nanstd,
                "q1": lambda x: nanquantile(a=x, q=0.25),
                "q3": lambda x: nanquantile(a=x, q=0.75),
                "trend_strength": lambda x: trend_strength(x, seasonal_period),
                "seasonal_strength": lambda x: seasonal_strength(x, seasonal_period),
                "lumpiness": lumpiness,
                "spikiness": spikiness,
                "curvature": curvature,
                "hurst_exponent": hurst_exponent,
                "spectral_entropy": spectral_entropy,
                "adf_pvalue": adf_pvalue,
            }
        )
    else:
        rolling_features = serie.rolling(window=seasonal_period).aggregate(
            {
                "mean": nanmean,
                "median": nanmedian,
                "std": nanstd,
                "q1": lambda x: nanquantile(a=x, q=0.25),
                "q3": lambda x: nanquantile(a=x, q=0.75),
                "trend_strength": lambda x: trend_strength(x, seasonal_period),
                "seasonal_strength": lambda x: seasonal_strength(x, seasonal_period),
                "lumpiness": lumpiness,
                "spikiness": spikiness,
                "curvature": curvature,
                "spectral_entropy": spectral_entropy,
                "adf_pvalue": adf_pvalue,
            }
        )
    # adding lags to the rolling features
    direct_lags = {f"lag {i}": serie.shift(i) for i in range(1, lags_to_consider + 1)}
    seasonal_lags = {
        f"seasonal lag {i}": serie.shift(i + seasonal_period)
        for i in range(1, lags_to_consider + 1)
    }

    for lag, seasonal_lag in zip(direct_lags.keys(), seasonal_lags.keys()):
        rolling_features[lag] = direct_lags[lag]
        rolling_features[seasonal_lag] = seasonal_lags[seasonal_lag]

    return rolling_features.dropna(axis=0)


def build_rolling_target(serie: ndarray, horizon: int) -> DataFrame:
    result = DataFrame({f"t+{i}": serie.shift(-i) for i in range(1, horizon + 1)})
    result.index = serie.index
    return result.dropna(axis=0)


def build_rolling_XY(
    serie: ndarray,
    seasonal_period: int,
    horizon: int = -1,
    lags_to_consider: int = 5,
) -> [DataFrame, DataFrame, DataFrame, DataFrame]:
    horizon = seasonal_period if horizon == -1 else horizon
    X = build_rolling_features(serie, horizon, lags_to_consider=5)
    y = build_rolling_target(serie, horizon)[lags_to_consider:]
    common_index = X.index.intersection(y.index)
    return X, X.loc[common_index], y, y.loc[common_index]


def temporal_train_test_split(
    X: ndarray, y: ndarray, test_size: float
) -> [ndarray, ndarray, ndarray, ndarray]:
    return (
        X[int(X.shape[0] * test_size) :],
        X[: int(X.shape[0] * test_size)],
        y[int(X.shape[0] * test_size) :],
        y[: int(X.shape[0] * test_size)],
    )
