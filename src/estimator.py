from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import RegressorMixin
from sklearn import metrics
from pandas import Series, DataFrame, date_range
from numpy import ndarray
from src.preprocessing_tools import build_rolling_XY
from src.plotting_tools import plot_rolling_features, plot_sequential_validation


class FeatureBasedEstimator(MultiOutputRegressor):
    ## Constructor
    def __init__(
        self,
        estimator: RegressorMixin,
        horizon: int = 1,
        seasonal_period: int = 12,
        freq: str = "D",
        n_jobs: int = None,
    ) -> None:
        super().__init__(estimator=estimator, n_jobs=n_jobs)
        self.horizon = horizon
        self.seasonal_period = seasonal_period
        self.freq = freq
        self.is_fitted = False

    ## Preprocessing, fit & forecast
    def fit(self, X: ndarray, y: ndarray):
        self.X = X
        self.y = y
        self.is_fitted = True
        return super().fit(self.X, self.y)

    def preprocess_and_fit(self, serie: Series, lags_to_consider: int = 5) -> None:
        self.rolling_features, self.X, _, self.y = build_rolling_XY(
            serie,
            seasonal_period=self.seasonal_period,
            horizon=self.horizon,
            lags_to_consider=lags_to_consider,
        )
        return self.fit(self.X, self.y)

    def forecast(self):
        if not (self.is_fitted):
            raise RuntimeError("Model need to be fitted to call this method.")

        preds = super().predict(self.rolling_features)[-1].ravel()
        return DataFrame(
            preds,
            index=date_range(
                start=max(self.rolling_features.index),
                periods=preds.shape[0] + 1,
                freq=self.freq,
            )[1:],
            columns=["Forecast"],
        )

    ## Sequential validation methods
    def __sequential_validation_splits(self, cv: int) -> [list, list]:
        # leaves one seasonal period out for the evaluation
        split_length = int(self.X.shape[0] / (cv + 1))
        return (
            [
                self.X[-i * split_length - self.seasonal_period : -self.seasonal_period]
                for i in range(1, cv + 1)
            ],
            [
                self.y[-i * split_length - self.seasonal_period : -self.seasonal_period]
                for i in range(1, cv + 1)
            ],
        )

    def sequential_validation(
        self, cv: int = 5, metric: metrics = metrics.mean_absolute_error
    ) -> dict:
        perfs = dict()
        if not self.is_fitted:
            raise RuntimeError("Model need to be fitted to call this method.")

        Xs, ys = self.__sequential_validation_splits(cv=cv)
        for x, y in zip(Xs, ys):
            preds = (
                super().fit(x, y).predict(self.X[-self.seasonal_period :])[-1].ravel()
            )
            perfs[f"{x.shape[0]}"] = metric(
                self.y.iloc[-1].values, preds[-self.seasonal_period :]
            )
        self.perfs = perfs
        self.metric = metric
        return perfs

    ## Plotting methods
    def plot_rolling_features(
        self, features_to_plot: list = None, save_path: str = None
    ) -> None:
        plot_rolling_features(self.rolling_features, features_to_plot, save_path)

    def plot_sequential_validation(self, cv: int = 5, save_path: str = None) -> None:
        perfs = self.sequential_validation(cv=cv)
        plot_sequential_validation(perfs, save_path, self.metric.__name__)

    ## Getters
    def get_horizon(self) -> int:
        return self.horizon

    def get_freq(self) -> str:
        return self.freq

    def get_seasonal_period(self) -> int:
        return self.seasonal_period

    def get_rolling_features(self) -> DataFrame:
        if self.is_fitted:
            return self.rolling_features
        else:
            raise RuntimeError("Model need to be fitted to call this method.")
