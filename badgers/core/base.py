import abc


class BaseTransformer:
    """
    
    """

    def fit(self, X, y=None, **fit_param):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it. Copied from https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

    @abc.abstractmethod
    def transform(self, X):
        pass
