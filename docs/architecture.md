# Architecture and key principles

## Core Module

The `core` module serves as the foundation of the Badgers framework, providing essential building blocks and infrastructure that other components rely on.

### Main Responsibilities:

1. **Base Classes**: Defines the fundamental `GeneratorMixin` abstract base class that all generators must inherit from, ensuring a consistent interface across the entire system.

2. **Standardized Interface**: Enforces a uniform `generate(X, y, **params)` method signature that returns transformed data `(Xt, yt)` for all generators.

3. **Input Preprocessing**: Provides decorator functions (`preprocess_inputs`) that automatically validate and convert input data to standardized formats (pandas DataFrames/Series).

4. **Pipeline Infrastructure**: Implements the `Pipeline` class that enables chaining multiple generators together in sequential workflows.

5. **Utility Functions**: Offers helper functions for common operations like probability normalization and random number generation.

## Generators Module

The `generators` module contains the actual implementation of various data transformation algorithms, organized by data type categories.

### Main Responsibilities:

1. **Data Transformation Implementation**: Houses concrete implementations of various data generation techniques across different data domains:
   - Tabular data transformations (outliers, drift, imbalance, missingness, noise)
   - Time series modifications (changepoints, seasons, trends, transmission errors)
   - Graph-based manipulations
   - Image processing generators
   - Text transformation tools

2. **Domain-Specific Organization**: Structures generators by data type categories, making it easy to find and use appropriate transformations for specific data modalities.

3. **Extensibility**: Provides a plug-and-play architecture where new generators can be easily added by following the established `GeneratorMixin` interface.