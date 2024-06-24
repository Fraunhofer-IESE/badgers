# Badgers: bad data generators

[badgers](https://github.com/Fraunhofer-IESE/badgers) is a python library for generating bad data (more precisely to augment existing data with data quality deficits such as outliers, missing values, noise, etc.). It is based upon a simple API and provides a set of generators object that can generate data quality deficits from existing data.

A word of caution: badgers is still in an early development stage. Although the core structure of the package and the `generate(X,y)` signature are not expected to change, some API details (like attributes names) are likely to change.


The full documentation is hosted here: [https://fraunhofer-iese.github.io/badgers/](https://fraunhofer-iese.github.io/badgers/).

For a quick-start, you can install `badgers` with pip:

```bash
pip install badgers
```

Import badgers as any other library and start using it:

```python
from sklearn.datasets import make_blobs
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator

X, y = make_blobs()
trf = GaussianNoiseGenerator(noise_std=0.5)
Xt, yt = trf.generate(X,y)
```

More examples are available in the [tutorials](https://fraunhofer-iese.github.io/badgers/tutorials/Imbalance-Tabular-Data/) section.

The API documentation is also available in the [API](https://fraunhofer-iese.github.io/badgers/reference/badgers/) section.

Interested developers will find relevant information in the [CONTRIBUTING.md](CONTRIBUTING.md) page. 