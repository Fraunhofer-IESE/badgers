# Getting started

Install `badgers` with pip:

```
pip install badgers
```

Import badgers as any other library and start using it:

```python
from sklearn.datasets import make_blobs
from scipy.stats import norm
from badgers.generators.tabular_data.noise import AdditiveNoiseGenerator

X, y = make_blobs()
gen = AdditiveNoiseGenerator()
Xt, yt = gen.generate(X, y, distribution=norm(scale=0.5))
```

More examples are available in the [tutorials](../tutorials/tabular-data/Imbalance-Tabular-Data/) section.

## Learn the concepts

New to badgers? The [Understanding badgers](../tutorials/concepts/Badgers-Concepts/) notebook explains the underlying objects and contracts **step by step**: library structure, how to instanciate generators, the `generate()` API, selectors, data-generating processes (DGP), pipelines and more.

## Build your own

Those interested in developing their own generators and/or contribute to badgers can have a look at the [dev-tutorials](../dev/Create-New-Tabular-Generators/) section — three notebooks:

* [Creating Tabular Generators](../dev/Create-New-Tabular-Generators/)
* [Creating Selectors](../dev/Create-New-Selectors/)
* [Creating DGP Backends](../dev/Create-New-DGP-Backends/)
* [Using DowhyGCM and PyMC](../dev/Using-DowhyGCM-And-PyMC/) — how the two bundled causal backends behave on fit / sample / intervene

For the big picture, see the [architecture](architecture.md) overview and [CONTRIBUTING.md](https://github.com/Fraunhofer-IESE/badgers/blob/main/CONTRIBUTING.md).

The API documentation is also available in the [API](../reference/badgers/) section.