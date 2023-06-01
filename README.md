# Badgers: bad data generators

[badgers](https://github.com/Fraunhofer-IESE/badgers) is a python library for generating bad data (more precisely to augment existing data with data quality deficits such as outliers, missing values, noise, etc.). It is based upon the scikit-learn API and provides a set of transformers object that can generate data quality deficits from existing data.

The full documentation is hosted here: [https://fraunhofer-iese.github.io/badgers/](https://fraunhofer-iese.github.io/badgers/).

For a quick-start, you can install `badgers` with pip:

```bash
pip install badgers
```

Import badgers as any other library and start using it:

```python
from sklearn.datasets import make_blobs
from badgers.transforms.tabular_data.noise import GaussianNoiseTransformer

X, y = make_blobs()
trf = GaussianNoiseTransformer(noise_std=0.5)
Xt = trf.transform(X)
```

More examples are available in the [tutorials](https://fraunhofer-iese.github.io/badgers/tutorials/Imbalance-Tabular-Data/) section.

The API documentation is also available in the [API](https://fraunhofer-iese.github.io/badgers/reference/badgers/) section.

Interested developers will find relevant information in the [CONTRIBUTING.md](CONTRIBUTING.md) page. 