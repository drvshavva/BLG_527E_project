import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Resources:
# 1. https://www.geeksforgeeks.org/building-a-custom-estimator-for-scikit-learn-a-comprehensive-guide/
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
# 3. https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

class OneClassGMM(BaseEstimator, ClassifierMixin):
    """
    Custom implementation since scikit-learn doesn't have a built-in One-Class GMM

    One-Class Gaussian Mixture Model (GMM) Classifier for anomaly detection.

    This estimator fits a Gaussian Mixture Model to the data and identifies
    anomalies based on the log-likelihood of each sample. Samples with low
    likelihood under the fitted GMM are considered anomalies.
    """

    def __init__(self, n_components=2, random_state=42):
        """
        n_components : int, default=2
            The number of mixture components in the GaussianMixture model.

        random_state : int, default=42
            Determines random number generation for initialization.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.threshold = None

    def fit(self, X, y=None):
        """  Fits the GMM to the training data and sets the anomaly threshold. """
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.gmm.fit(X)
        # Getting scores for train dataset (for threshold)
        scores = self.gmm.score_samples(X)
        # The log-likelihood threshold below which samples are considered anomalies
        # default is the 5th percentile of the training scores

        self.threshold = np.percentile(scores, 5)
        return self

    def decision_function(self, X):
        """Return decision scores (log-likelihood)"""
        if self.gmm is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.gmm.score_samples(X)

    def predict(self, X):
        """Predict if samples are outliers (1 for inliers, -1 for outliers)"""
        scores = self.decision_function(X)
        return np.where(scores >= self.threshold, 1, -1)