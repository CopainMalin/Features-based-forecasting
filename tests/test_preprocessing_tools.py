import unittest
from pandas import read_csv, to_datetime
from numpy.random import choice
from numpy import zeros
from src.preprocessing_tools import (
    build_rolling_features,
    build_rolling_target,
    temporal_train_test_split,
)


class TestFeaturesBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # ~1 min to set Up
        # sourcery skip: class-method-first-arg-name
        serie = read_csv(
            "./datas/test_dataset.csv", sep=",", index_col="time_stamp", nrows=1000
        )
        serie.index = to_datetime(serie.index)
        serie = serie.loc[:, choice(serie.columns)]

        cls.serie = serie
        cls.seasonal_period = 365
        cls.nb_lags_to_consider = 6
        cls.features_list = [
            "mean",
            "median",
            "std",
            "q1",
            "q3",
            "seasonal_strength",
            "trend_strength",
            "spikiness",
            "lumpiness",
            "curvature",
            "spectral_entropy",
            "adf_pvalue",
        ]
        for i in range(1, cls.nb_lags_to_consider):
            cls.features_list.append(f"lag {i}")
            cls.features_list.append(f"seasonal lag {i}")

        cls.builded_features = build_rolling_features(cls.serie, cls.seasonal_period)

    def test_all_features_are_computed(self):
        self.assertTrue(
            set((self.features_list)).issubset(self.builded_features.columns),
            msg=f"Missing some features after the features building.\nMissing : {[x for x in self.features_list if x not in self.builded_features.columns]}",
        )

    def test_nb_rolling_obs(self):
        self.assertTrue(
            self.builded_features.shape[0]
            == self.serie.shape[0]
            - self.seasonal_period
            + 1
            - self.nb_lags_to_consider,
            msg=f"Expected: {self.serie.shape[0] - self.seasonal_period + 1 - self.nb_lags_to_consider} values.\nGot: {self.builded_features.shape[0]} values.",
        )


class TestTargetBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serie = read_csv(
            "./datas/test_dataset.csv", sep=",", index_col="time_stamp", nrows=1000
        )
        serie.index = to_datetime(serie.index)
        serie = serie.loc[:, choice(serie.columns)]
        cls.serie = serie
        cls.seasonal_period = 365

    def test_target_build_dim(self):
        builded_target = build_rolling_target(self.serie, self.seasonal_period)
        self.assertEqual(
            builded_target.shape[0],
            self.serie.shape[0] - self.seasonal_period,
            msg=f"Wrong number of rows.\nExpected: {self.serie.shape[0] - self.seasonal_period}.\nGot: {builded_target.shape[0]}.",
        )
        self.assertEqual(
            builded_target.shape[1],
            self.seasonal_period,
            msg=f"Wrong number of columns.\nExpected: {self.seasonal_period}.\nGot: {builded_target.shape[1]}.",
        )


class TestSplit(unittest.TestCase):
    def test_temporal_train_test_split(self):
        X = zeros((100, 5))
        y = zeros(100)
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            X, y, test_size=0.33
        )
        self.assertEqual(
            X_test.shape[0],
            int(X.shape[0] * 0.33),
            msg="Test set does not have the expected size.",
        )
        self.assertEqual(
            X_train.shape[1],
            X_test.shape[1],
            msg="Train and test features matrices does not have the same number of features.",
        )
        self.assertEqual(
            y_train.shape[0],
            X_train.shape[0],
            msg="Train features and target does not have the same length.",
        )
