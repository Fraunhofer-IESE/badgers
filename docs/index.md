# Welcome to Badgers

Badgers is a python library for generating bad data (more precisely to augment existing data with data quality deficits such as outliers, missing values, noise, etc.).



It is based upon the scikit-learn API. Most objects provided by badgers are transformers (i.e., they provide a `transform()` and/or a `fit_transform()` function) and should be compatible with any framework that supports or use the scikit-learn API.


Why would you generate bad data? you might ask (and you should! isn't that honestly a strange idea?).

We think data quality has to be taken seriously. With [badgers](https://github.com/Fraunhofer-IESE/badgers) we hope to provide a tool that can help manage and understand the impact of data quality in a systematic and controlled way.

You might think of using badgers for things like robustness analysis (i.e., how does my model or my data analysis pipeline performs in the presence of noise, outliers, missing values, data drift, etc.), or for chaos data engineering (e.g., what happens if we inject quality defects into production systems?).



