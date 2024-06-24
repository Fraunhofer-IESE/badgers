# Welcome to Badgers

[Badgers](https://github.com/Fraunhofer-IESE/badgers) is a python library for generating bad data - more precisely: to augment existing data with data quality deficits such as outliers, missing values, noise, etc.

As a basic principle, [badgers](https://github.com/Fraunhofer-IESE/badgers) provides a set of objects (called generators) that follow a simple API: each generator provides a `generator(X,y,**params)` function that takes as argument `X` (the input features), `y` (the class labels, the regression target, or None), a set of specific parameters and returns the transformed `Xt` and `yt`.

Why would you generate bad data? you might ask (and you should! isn't that honestly a strange idea?).

We think data quality has to be taken seriously. With [badgers](https://github.com/Fraunhofer-IESE/badgers) we hope to provide a tool that can help manage and understand the impact of data quality in a systematic and controlled way.

You might think of using [badgers](https://github.com/Fraunhofer-IESE/badgers) for things like robustness analysis (i.e., how does my model or my data analysis pipeline performs in the presence of noise, outliers, missing values, data drift, etc.), or for chaos data engineering (e.g., what happens if we inject quality defects into production systems?).


[Badgers](https://github.com/Fraunhofer-IESE/badgers) provides a set of predefined generators for different modalities (tabular data, time series, text, etc.) and different data quality problems (outliers, noise, drift, etc.). Of course many data quality problems are use case dependent and it is not possible to implement all generators. Therefore, the idea is that badgers can serve as a structure for developing novel generators (see how to develop novel generators in the [dev-tutorials](../tutorials/Create-New-Tabular-Generators/) section).

 Want to try [badgers](https://github.com/Fraunhofer-IESE/badgers)? Then go to the [getting started](getting-started.md) section or dive into the [tutorials](../tutorials/Imbalance-Tabular-Data/) section.



