# Tutorials

A guided tour of badgers through its deficits. All tutorials are Jupyter notebooks.

If you are new to badgers, start with the [Understanding badgers](concepts/Badgers-Concepts/) concepts notebook — it explains generators, selectors, DGPs and pipelines step by step before you dive into the per-deficit tutorials.

## Concepts

| Goal | Tutorial |
|---|---|
| Understand the building blocks (generators, selectors, DGPs, pipelines) | [Understanding badgers](concepts/Badgers-Concepts/) |

## Tabular data

| Goal | Tutorial |
|---|---|
| Add outliers to tabular data | [Generating Outliers](tabular-data/Outliers/) |
| Add noise | [Generating Noise](tabular-data/Noise/) |
| Introduce missing values | [Generating Missing Data](tabular-data/Missingness/) |
| Generate class imbalance | [Generating Imbalanced Data](tabular-data/Imbalance/) |
| Shift data over time (drift) | [Generating Drift](tabular-data/Drift/) |

## Beyond tabular data

| Goal | Tutorial |
|---|---|
| Corrupt text data (typos) | [Generating Typos](text/Typos-Text/) |
| Corrupt time series | the *Time Series Data* tutorials below |

## Time series data

| Goal | Tutorial |
|---|---|
| Generate point outliers | [Generating Point Outliers](time-series/Outliers-Time-Series/) |
| Introduce missing values | [Generating Missing Data](time-series/Missingness-Time-Series/) |
| Add noise | [Generating Noise](time-series/Noise-Time-Series/) |
| Generate patterns | [Generating Patterns](time-series/Patterns-Time-Series/) |
| Generate trends | [Generating Trends](time-series/Trends-Time-Series/) |
| Generate seasons | [Generating Seasons](time-series/Seasons-Time-Series/) |
| Generate changepoints | [Generating Changepoints](time-series/Changepoints-Time-Series/) |
| Generate "transmission errors" | [Generating "Transmissions Errors"](time-series/Transmission-Errors-Time-Series/) |

## Pipelines and developing your own

| Goal | Tutorial |
|---|---|
| Chain several deficits | [Chaining Generators](Pipeline-Tabular-Data/) |
| Build custom generators | [Creating Tabular Generators](dev/Create-New-Tabular-Generators/) |
| Build custom selectors | [Creating Selectors](dev/Create-New-Selectors/) |
| Use the bundled causal backends (dowhy-gcm / pymc) | [Using DGPs](dev/Using-DGPs/) |
