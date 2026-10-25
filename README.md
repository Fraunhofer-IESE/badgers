# Badgers: bad data generators

[badgers](https://github.com/Fraunhofer-IESE/badgers) is a python library for generating bad data (more precisely to augment existing data with data quality deficits such as outliers, missing values, noise, etc.). It is based upon a simple API and provides a set of generators object that can generate data quality deficits from existing data.

A word of caution: badgers is still in an early development stage. The core structure of the package and the `generate(X, **params)` signature — generators and selectors take the input data `X` only and return a single generated object — are not expected to change, but some API details (like attribute names) are likely to change. Target-like data (class labels, signals used for selection) lives inside `X` as dedicated columns (or graph attributes / modality-native values for other data types).


The full documentation is hosted here: [https://fraunhofer-iese.github.io/badgers/](https://fraunhofer-iese.github.io/badgers/).

For a quick-start, you can install `badgers` with pip:

```bash
pip install badgers
```

Import badgers as any other library and start using it:

```python
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from scipy.stats import norm
from badgers.generators.tabular_data.noise import AdditiveNoiseGenerator

X, _ = make_blobs()
trf = AdditiveNoiseGenerator(random_generator=default_rng(0))

Xt = trf.generate(X, distribution=norm(scale=0.5))
```

More examples are available in the [tutorials](https://fraunhofer-iese.github.io/badgers/tutorials/tabular-data/Imbalance/) section.

The API documentation is also available in the [API](https://fraunhofer-iese.github.io/badgers/reference/badgers/) section.

Interested developers will find relevant information in the [CONTRIBUTING.md](CONTRIBUTING.md) page. 
