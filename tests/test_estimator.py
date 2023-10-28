import unittest
from sklearn.linear_model import LinearRegression
from src.estimator import FeatureBasedEstimator
from numpy import zeros


class TestEstimator(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        cls.estimator_ = FeatureBasedEstimator(
            estimator=LinearRegression, horizon=10, seasonal_period=12, freq="H"
        )

    def test_getters(self):
        self.assertEqual(self.estimator_.get_horizon(), 10)
        self.assertEqual(self.estimator_.get_seasonal_period(), 12)
        self.assertEqual(self.estimator_.get_freq(), "H")
        self.assertEqual(self.estimator_.is_fitted, False)
