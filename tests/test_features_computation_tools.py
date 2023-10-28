import unittest
from numpy import (
    nanmean,
    nanmedian,
    nanstd,
    sin,
    nan,
    sqrt,
    pi,
    nanquantile,
    abs,
)  # Maths fuctions
from numpy import array, zeros, arange  # Structure / data generation
from numpy.testing import assert_equal
from numpy.random import randn

from hurst import random_walk

from src.features_computation_tools import (
    seasonal_strength,
    trend_strength,
    spikiness,
    lumpiness,
    curvature,
    spectral_entropy,
    hurst_exponent,
    autocorrelation,
    partial_autocorrelation,
    adf_pvalue,
)


# Numpy functions that can be used for the feature based forecasting
class TestNumpy(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(
            nanmean([1, 2, 3, 4, 5]),
            (1 + 2 + 3 + 4 + 5) / 5,
            msg="Computation of the mean uncorrect.",
        )

    def test_std(self):
        self.assertAlmostEqual(
            nanstd([1, 2, 3, 4, 5]),
            sqrt(nanmean(([1, 2, 3, 4, 5] - nanmean([1, 2, 3, 4, 5])) ** 2)),
            msg="Computation of the standard deviation uncorrect.",
        )

    def test_q1(self):
        self.assertEqual(
            nanquantile(a=[1, 2, 3, 4, 5, 6], q=0.25),
            2.25,
            msg="Computation of the first quartile.",
        )

    def test_median(self):
        self.assertEqual(
            nanquantile(a=[1, 2, 3, 4, 5, 6], q=0.5),
            3.5,
            msg="Computation of the median uncorrect (0.5 quantile function test).",
        )
        self.assertEqual(
            nanmedian([1, 2, 3, 4, 5, 6]),
            3.5,
            msg="Computation of the median uncorrect (median function test).",
        )

    def test_q3(self):
        self.assertEqual(
            nanquantile(a=[1, 2, 3, 4, 5, 6], q=0.75),
            4.75,
            msg="Computation of the mean uncorrect.",
        )


# Other features used for the forecasting. One class per feature.
class TestSeasonalStrenght(unittest.TestCase):
    def test_strong_seasonal_case(self) -> None:
        mock_data = 3 * sin(2 * pi * arange(1, 101) / 10)
        self.assertAlmostEqual(
            seasonal_strength(mock_data, 10),
            1,
            msg="Strongly seasonal serie should give a seasonal strenght result close to 1.",
        )

    def test_weak_seasonal_case(self) -> None:
        self.assertAlmostEqual(
            seasonal_strength(zeros(100), 10),
            0,
            msg="Non-seasonal serie should give a seasonal strenght result close to 0.",
        )

    def test_type_check(self):
        self.assertIsInstance(
            seasonal_strength(randn(100), 10),
            float,
            msg="seasonal_strenght should return a float.",
        )


class TestTrendStrenght(unittest.TestCase):
    def test_trended_case(self) -> None:
        self.assertAlmostEqual(
            trend_strength(arange(100), 10),
            1,
            msg="Strongly trended serie should give a trend strenght result close to 1.",
        )

    def test_untrended_case(self) -> None:
        self.assertAlmostEqual(
            trend_strength(zeros(100), 10),
            0,
            msg="Untrended serie should give a trend strenght result close to 1.",
        )

    def test_type_check(self):
        self.assertIsInstance(
            trend_strength(randn(100), 10),
            float,
            msg="trend_strength should return a float.",
        )


class TestSpikiness(unittest.TestCase):
    def test_unspiked_data(self) -> None:
        self.assertEqual(
            spikiness(zeros(100)),
            0,
            msg="Unspiked serie should give a spikiness result of 0.",
        )

    def test_two_spiked_data(self) -> None:
        mock_data = zeros(100)
        mock_data[0] = mock_data[1] = 10
        self.assertEqual(
            spikiness(mock_data),
            0.02,
            msg="A serie of 100 values with 2 spikes should give a spikiness of 0.2.",
        )

    def type_check(self):
        self.assertIsInstance(
            spikiness(randn(100)),
            float,
            msg="spikiness should return a float.",
        )


class TestLumpiness(unittest.TestCase):
    def test_lumpiness(self) -> None:
        mock_data = zeros(100)
        mock_data[0] = 1
        self.assertTrue(
            expr=(lumpiness(mock_data) > 0)
            & (lumpiness(mock_data) < mock_data.shape[0]),
            msg="Lumpiness should be greater than zero and lesser than 100 in this particular case.",
        )

    def type_check(self):
        self.assertIsInstance(
            lumpiness(randn(100)),
            float,
            msg="lumpiness should return a float.",
        )


class TestCurvature(unittest.TestCase):
    def test_no_curvature(self) -> None:
        self.assertEqual(
            0,
            curvature(zeros(100)),
            msg="A serie of zeros should return a curvature of zeros.",
        )

    def test_curvature(self) -> None:
        self.assertAlmostEqual(
            curvature(array([2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7])),
            12.4,
            msg="Curvature should be around 12.4 in this particular case.",
        )

    def test_type_check(self):
        self.assertIsInstance(
            curvature(randn(100)), float, msg="curvature should return a float."
        )


class TestSpectralEntropy(unittest.TestCase):
    def test_uniform_case(self):
        assert_equal(
            spectral_entropy(zeros(100)),
            nan,
            err_msg="Spectral entropy of a constant value must be NaN (or 0).",
        )

    def test_other_cases(self):
        self.assertGreater(
            spectral_entropy(randn(100)),
            0,
            msg="Spectral entropy of a random vector should be greater than 0.",
        )

    def type_check(self):
        self.assertIsInstance(
            spectral_entropy(randn(100)),
            float,
            msg="spectral_entropy should return a float.",
        )


class TestHurstExponent(unittest.TestCase):
    def test_mean_revertic_case(self):
        self.assertLess(
            hurst_exponent(random_walk(10000, proba=0.3)),
            0.4,
            msg="In mean revertic cases, Hurst exponent should be at least lesser than 0.4.",
        )

    def test_random_case(self):
        brownian = random_walk(10000, proba=0.5)
        self.assertLessEqual(
            hurst_exponent(brownian),
            0.6,
            msg="In brownian motion cases, Hurst exponent must be lesser than 0.6.",
        )
        self.assertGreaterEqual(
            hurst_exponent(brownian),
            0.4,
            msg="In brownian motion cases, Hurst exponent must be upper than 0.4.",
        )

    def test_persistance(self):
        self.assertGreater(
            hurst_exponent(random_walk(10000, proba=0.7)),
            0.6,
            msg="In persistents cases, Hurst exponent must be greater than 0.6.",
        )

    def test_type_check(self):
        self.assertIsInstance(
            hurst_exponent(random_walk(10000, proba=0.7)),
            float,
            msg="hurst_exponent should return a float.",
        )


class TestACFs(unittest.TestCase):
    def test_lower_boud(self):
        [
            (
                self.assertGreaterEqual(
                    int(element), 0, msg="ACF should be greater than 0."
                )
            )
            for element in autocorrelation(randn(100), 10)
        ]

    def test_upper_boud(self):
        [
            (self.assertLessEqual(int(element), 1, msg="ACF should be lesser than 1."))
            for element in autocorrelation(arange(100), 10)
        ]

    def test_type_check(self):
        [
            self.assertIsInstance(
                element, float, msg="ACF results must be an iterable of floats."
            )
            for element in autocorrelation(randn(100), 10)
        ]


class TestPACFs(unittest.TestCase):
    def test_lower_boud(self):
        [
            (
                self.assertGreaterEqual(
                    int(element), 0, msg="PACF should be greater than 0."
                )
            )
            for element in partial_autocorrelation(randn(100), 10)
        ]

    def test_upper_boud(self):
        [
            (self.assertLessEqual(int(element), 1, msg="PACF should be lesser than 1."))
            for element in partial_autocorrelation(arange(100), 10)
        ]

    def test_type_check(self):
        [
            self.assertIsInstance(
                element, float, msg="PACF results must be an iterable of floats."
            )
            for element in partial_autocorrelation(randn(100), 10)
        ]


class TestUnitRoot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # H0 : No unit root
        # H1 : Unit root
        noise = randn(100)
        serie_with_unit_root = [1]
        serie_without_unit_root = [1]
        for i in range(1, 100):
            serie_with_unit_root.append(serie_with_unit_root[i - 1] * 1.3 + noise[i])
            serie_without_unit_root.append(
                serie_without_unit_root[i - 1] * 0.7 + noise[i]
            )

        cls.serie_with_unit_root = array(serie_with_unit_root)
        cls.serie_without_unit_root = array(serie_without_unit_root)

    def test_no_unit_root_series(self):
        self.assertGreaterEqual(
            0.05,
            adf_pvalue(self.serie_without_unit_root),
            msg="No unit root cases should return a pvalue >= 0.05.",
        )

    def test_unit_root_series(self):
        self.assertLessEqual(
            0.05,
            adf_pvalue(self.serie_with_unit_root),
            msg="Unit root cases should return a pvalue >= 0.05.",
        )


if __name__ == "__main__":
    unittest.main()
