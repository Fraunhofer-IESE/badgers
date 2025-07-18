import abc
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs
from badgers.generators.time_series.utils import get_patterns_uniform_probability


class TransmissionErrorGenerator(GeneratorMixin):
    """
    Base class for transformers that generate transmission errors

    Transmission errors are errors that includes values dropped, delay, switching values over time...
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the TransmissionErrorGenerator with a specified random number generator.

        :param random_generator: A random number generator instance (default is a NumPy random generator seeded with 0).
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Abstract method to generate transmission errors on the input data.

        :param X: Input features, expected to be a pandas DataFrame.
        :param y: Target variable, expected to be a pandas Series.
        :param params: Additional parameters that might be needed for generating errors.
        :return: A tuple containing the modified input features and target variable with transmission errors applied.
        """
        pass


class RandomTimeSwitchGenerator(TransmissionErrorGenerator):
    """
    Switches time randomly (for now uniformly at random).
    A time series x(t) = {x(0), x(1), x(2), ..., x(t), x(t+1), ..., x(n)} is transformed to
                 x'(t) = {x(0), x(1), x(2), ..., x(t+1), x(t), ..., x(n)}
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the RandomTimeSwitchGenerator with a specified random number generator.

        :param random_generator: A random number generator instance (default is a NumPy random generator seeded with 0).
        """
        super().__init__(random_generator=random_generator)
        self.switch_indices_ = None

    @preprocess_inputs
    def generate(self, X, y, n_switches: int = 10) -> Tuple:
        """
        Introduces `n_switches` random switches in the input time series data X.

        This method randomly selects `n_switches` pairs of consecutive indices (i, i+1) and swaps their values in X.
        The target variable y remains unchanged.

        :param X: A pandas DataFrame representing the input time series data.
        :param y: A pandas Series representing the target variable (remains unchanged).
        :param n_switches: An integer specifying the number of random switches to introduce in X.
        :return: A tuple (Xt, y) where Xt is the modified input time series data with random switches applied,
                 and y is the original target variable.
        """
        assert n_switches > 0, 'n_switches should be strictly greater than 0'

        self.switch_indices_ = self.random_generator.choice(len(X) - 1, size=n_switches, replace=False)
        for i in self.switch_indices_:
            tmp = X.iloc[i].copy()
            X.iloc[i] = X.iloc[i + 1].copy()
            X.iloc[i + 1] = tmp
        return X, y


class RandomRepeatGenerator(TransmissionErrorGenerator):
    """
    Repeats randomly values

    a time series x(t) = {x(0), x(1), x(2), x(3), x(4), x(5)} would be transformed to
                 x'(t) = {x(0), x(1), x(1), x(2), x(3), x(4), x(4), x(4), x(5)}

    This simulates a delay in transmission where several values are repeated over time
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the RandomRepeatGenerator with a specified random number generator.

        :param random_generator: A random number generator instance (default is a NumPy random generator seeded with 0).
        """
        super().__init__(random_generator=random_generator)
        self.repeats_ = None  # to store the indices of the repeats (from the original X) and the length of the repeat

    @preprocess_inputs
    def generate(self, X, y, n_repeats: int = 10, min_nb_repeats: int = 1,
                 max_nb_repeats: int = 10) -> Tuple:
        """
        Introduces `n_repeats` random repetitions in the input time series data X.

        This method randomly selects `n_repeats` indices from X and repeats each selected value a random number of times between
        `min_nb_repeats` and `max_nb_repeats`. The repeated values are inserted immediately after the selected index in X.
        The target variable y remains unchanged.

        :param X: A pandas DataFrame representing the input time series data.
        :param y: A pandas Series representing the target variable (remains unchanged).
        :param n_repeats: An integer specifying the number of random repetitions to introduce in X.
        :param min_nb_repeats: An integer specifying the minimum number of times a value can be repeated.
        :param max_nb_repeats: An integer specifying the maximum number of times a value can be repeated.
        :return: A tuple (Xt, y) where Xt is the modified input time series data with random repetitions applied,
                 and y is the original target variable.
        """
        assert n_repeats > 0, 'n_repeats should be strictly greater than 0'

        # generate indices for repeats
        self.repeats_ = [(i, l) for i, l in zip(
            sorted(self.random_generator.choice(len(X), size=n_repeats, replace=False, shuffle=False)),
            self.random_generator.integers(low=min_nb_repeats, high=max_nb_repeats, size=n_repeats)
        )]

        Xt = X.values

        # generate the repeats and insert them
        offset = 0

        for i, l in self.repeats_:
            repeat = np.repeat(X.iloc[i], l)
            Xt = np.insert(Xt, i + offset, repeat)
            offset += l

        # create a pandas dataframe with same columns as X
        Xt = pd.DataFrame(data=Xt, columns=X.columns)

        return Xt, y


class RandomDropGenerator(TransmissionErrorGenerator):
    """
    Drops randomly values

    a time series x(t) = {x(0), x(1), x(2), x(3), x(4), x(5)} would be transformed to
                 x'(t) = {x(0), x(1), x(3), x(4)}

    This simulates a problem in transmission where several values are randomly dropped
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the RandomDropGenerator with a specified random number generator.

        :param random_generator: A random number generator instance (default is a NumPy random generator seeded with 0).
        """
        super().__init__(random_generator=random_generator)
        self.drops_indices_ = None  # to store the indices of the drops

    @preprocess_inputs
    def generate(self, X, y, n_drops: int = 10) -> Tuple:
        """
        Introduces `n_drops` random drops in the input time series data X.

        This method randomly selects `n_drops` indices from X and removes the corresponding rows.
        The target variable y remains unchanged.

        :param X: A pandas DataFrame representing the input time series data.
        :param y: A pandas Series representing the target variable (remains unchanged).
        :param n_drops: An integer specifying the number of random drops to introduce in X.
        :return: A tuple (Xt, y) where Xt is the modified input time series data with random drops applied,
                 and y is the original target variable.
        """
        assert n_drops > 0, 'n_drops should be strictly greater than 0'

        # generate indices for drops
        self.drops_indices_ = self.random_generator.choice(len(X), size=n_drops, replace=False, shuffle=False)

        Xt = X.drop(X.index[self.drops_indices_], axis=0).reset_index(drop=True)

        return Xt, y


class LocalRegionsRandomDropGenerator(TransmissionErrorGenerator):
    """
    Drops randomly values in specific regions of time

    This simulates a problem in transmission where several values are randomly dropped at specific time intervals (time regions)
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the LocalRegionsRandomDropGenerator with a specified random number generator.

        :param random_generator: A random number generator instance (default is a NumPy random generator seeded with 0).
        """
        super().__init__(random_generator=random_generator)
        self.drops_indices_ = None  # to store the indices of the drops
        self.drops_probabilities_ = None # to store the probability of dropping values

    @preprocess_inputs
    def generate(self, X, y, n_drops: int = 10, n_regions: int = 5, min_width_regions: int = 5,
                 max_width_regions: int = 10) -> Tuple:
        """
        Introduces `n_drops` random drops in the input time series data X within `n_regions` specific time regions.

        This method randomly defines `n_regions` time regions within the time series, each having a width between
        `min_width_regions` and `max_width_regions`. Within each region, values are randomly dropped until the total number of
        dropped values reaches `n_drops`. The target variable y remains unchanged.

        :param X: A pandas DataFrame representing the input time series data.
        :param y: A pandas Series representing the target variable (remains unchanged).
        :param n_drops: An integer specifying the total number of random drops to introduce in X.
        :param n_regions: An integer specifying the number of time regions where values will be dropped.
        :param min_width_regions: An integer specifying the minimum width of the time regions (intervals).
        :param max_width_regions: An integer specifying the maximum width of the time regions (intervals).
        :return: A tuple (Xt, y) where Xt is the modified input time series data with random drops applied within specific
                 time regions, and y is the original target variable.
        """
        assert n_drops > 0, 'n_drops should be strictly greater than 0'

        # generate probability for where to drop points
        self.drops_probabilities_ = get_patterns_uniform_probability(
            signal_size=len(X),
            random_generator=self.random_generator,
            n_patterns=n_regions,
            min_width_pattern=min_width_regions,
            max_width_patterns=max_width_regions
        )

        # generate indices for drops
        self.drops_indices_ = self.random_generator.choice(len(X), size=n_drops, p=self.drops_probabilities_, replace=False, shuffle=False)

        Xt = X.drop(X.index[self.drops_indices_], axis=0).reset_index(drop=True)

        return Xt, y


class LocalRegionsRandomRepeatGenerator(TransmissionErrorGenerator):
    """
    Repeats randomly values x(t) only in certain time regions (time intervals)

    This simulates a delay in transmission where several values are repeated over time (in certain time intervals)
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the LocalRegionsRandomRepeatGenerator with a specified random number generator.

        :param random_generator: A random number generator instance (default is a NumPy random generator seeded with 0).
        """
        super().__init__(random_generator=random_generator)
        self.repeats_ = None  # to store the indices of the repeats (from the original X) and the length of the repeat
        self.repeats_probabilities_ = None  # to store the probability of repeating values

    @preprocess_inputs
    def generate(self, X, y, n_repeats: int = 10, min_nb_repeats: int = 1,
                 max_nb_repeats: int = 10, n_regions: int = 5, min_width_regions: int = 5,
                 max_width_regions: int = 10) -> Tuple:
        """
        Introduces `n_repeats` random repetitions in the input time series data X within `n_regions` specific time regions.

        This method randomly defines `n_regions` time regions within the time series, each having a width between
        `min_width_regions` and `max_width_regions`. Within each region, values are randomly selected and repeated a random number of times
        between `min_nb_repeats` and `max_nb_repeats`. The repeated values are inserted immediately after the selected index in X.
        The target variable y remains unchanged.

        :param X: A pandas DataFrame representing the input time series data.
        :param y: A pandas Series representing the target variable (remains unchanged).
        :param n_repeats: An integer specifying the total number of random repetitions to introduce in X.
        :param min_nb_repeats: An integer specifying the minimum number of times a value can be repeated.
        :param max_nb_repeats: An integer specifying the maximum number of times a value can be repeated.
        :param n_regions: An integer specifying the number of time regions where values will be repeated.
        :param min_width_regions: An integer specifying the minimum width of the time regions (intervals).
        :param max_width_regions: An integer specifying the maximum width of the time regions (intervals).
        :return: A tuple (Xt, y) where Xt is the modified input time series data with random repetitions applied within specific
                 time regions, and y is the original target variable.
        """
        assert n_repeats > 0, 'n_repeats should be strictly greater than 0'

        # generate probability for where to drop points
        self.repeats_probabilities_ = get_patterns_uniform_probability(
            signal_size=len(X),
            random_generator=self.random_generator,
            n_patterns=n_regions,
            min_width_pattern=min_width_regions,
            max_width_patterns=max_width_regions
        )

        # generate indices for repeats
        self.repeats_ = [(i, l) for i, l in zip(
            sorted(self.random_generator.choice(len(X), size=n_repeats, p=self.repeats_probabilities_, replace=False, shuffle=False)),
            self.random_generator.integers(low=min_nb_repeats, high=max_nb_repeats, size=n_repeats)
        )]

        Xt = X.values

        # generate the repeats and insert them
        offset = 0

        for i, l in self.repeats_:
            repeat = np.repeat(X.iloc[i], l)
            Xt = np.insert(Xt, i + offset, repeat)
            offset += l

        # create a pandas dataframe with same columns as X
        Xt = pd.DataFrame(data=Xt, columns=X.columns)

        return Xt, y