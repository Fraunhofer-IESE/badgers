from unittest import TestCase

import numpy as np
import pandas as pd
from badgers.generators.time_series.trends import TrendsGenerator

from badgers.generators.time_series.seasons import SeasonsGenerator

from badgers.generators.time_series.patterns import PatternsGenerator

from badgers.generators.tabular_data.outliers import OutliersGenerator
from badgers.generators.time_series.changepoints import ChangePointsGenerator

from badgers.generators.tabular_data.noise import NoiseGenerator


class TestTimeSeriesFormat(TestCase):

    def setUp(self):
        size = 100
        self.format_to_tests = {
            'list': [0] * size,
            'narray_1_dim': np.zeros(size),
            'narray_1_dim_reshape': np.zeros(size).reshape(-1, 1),
            'narray_2_dim': np.zeros((size, 2)),
            'pd_series': pd.Series(data=np.zeros(size)),
            'pd_dataframe': pd.DataFrame(data=np.zeros(size).reshape(-1, 1))
        }

    def test_noise_generators_with_different_formats(self):
        size = 100

        for generator_cls in NoiseGenerator.__subclasses__():
            generator = generator_cls()
            for format_name, X in self.format_to_tests.items():
                with self.subTest(format_name=format_name):
                    generator.generate(X=X, y=None)

    def test_changepoints_generators_with_different_formats(self):
        size = 100

        for generator_cls in ChangePointsGenerator.__subclasses__():
            generator = generator_cls()
            for format_name, X in self.format_to_tests.items():
                with self.subTest(format_name=format_name):
                    generator.generate(X=X, y=None)

    def test_outliers_generators_with_different_formats(self):
        size = 100

        for generator_cls in OutliersGenerator.__subclasses__():
            generator = generator_cls()
            for format_name, X in self.format_to_tests.items():
                with self.subTest(format_name=format_name):
                    generator.generate(X=X, y=None)

    def test_patterns_generators_with_different_formats(self):
        size = 100

        for generator_cls in PatternsGenerator.__subclasses__():
            generator = generator_cls()
            for format_name, X in self.format_to_tests.items():
                with self.subTest(format_name=format_name):
                    generator.generate(X=X, y=None)

    def test_seasons_generators_with_different_formats(self):
        size = 100

        for generator_cls in SeasonsGenerator.__subclasses__():
            generator = generator_cls()
            for format_name, X in self.format_to_tests.items():
                with self.subTest(format_name=format_name):
                    generator.generate(X=X, y=None)

    def test_trends_generators_with_different_formats(self):
        size = 100

        for generator_cls in TrendsGenerator.__subclasses__():
            generator = generator_cls()
            for format_name, X in self.format_to_tests.items():
                with self.subTest(format_name=format_name):
                    generator.generate(X=X, y=None)