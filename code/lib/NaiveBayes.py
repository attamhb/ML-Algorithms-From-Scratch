import numpy as np


class NaiveBayesClassifier:

    def fit(self, features, labels):
        n_samples, n_features = features.shape
        self.classes = np.unique(labels)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            features_cls = features[labels == cls]
            self.means[idx, :] = features_cls.mean(axis=0)
            self.variances[idx, :] = features_cls.var(axis=0)
            self.priors[idx] = features_cls.shape[0] / float(n_samples)

    def predict(self, test_features):
        predictions = [
            self._predict_single(sample) for sample in test_features
        ]
        return np.array(predictions)

    def _predict_single(self, sample):
        posteriors = []

        for idx, cls in enumerate(self.classes):
            log_prior = np.log(self.priors[idx])
            log_posterior = np.sum(
                np.log(self._probability_density_function(idx, sample))
            )
            total_posterior = log_posterior + log_prior
            posteriors.append(total_posterior)

        return self.classes[np.argmax(posteriors)]

    # Gaussian Distribution
    def _probability_density_function(self, class_idx, sample):
        mean = self.means[class_idx]
        variance = self.variances[class_idx]
        return np.exp(-((sample - mean) ** 2) / (2 * variance)) / np.sqrt(
            2 * np.pi * variance
        )

    # beroulli distribution
    def _bernoulli_probability(self, x_d, theta_dc):
        return theta_dc if x_d == 1 else (1 - theta_dc)

    # beroulli distribution
    def _p_x_given_y_binary(self, x, y_c, theta):
        D = len(x)
        probability = 1.0

        for d in range(D):
            # Assume theta is a 2D array where theta[d][y_c] gives theta_dc
            theta_dc = theta[d][y_c]
            probability *= self._bernoulli_probability(x[d], theta_dc)

        return probability

    def _categorical_probability(self, x_d, theta_dc):
        """Compute the categorical probability for a single x_d given theta_dc."""
        return theta_dc[x_d - 1]  # Adjust for 0-based index

    def _p_x_given_y_catorical(self, x, y_c, theta):
        """Compute the probability p(x | y=c, theta) as the product of categorical distributions."""
        D = len(x)  # Number of components
        probability = 1.0

        for d in range(D):
            # Access the appropriate theta value for the observation and class
            theta_dc = theta[d][y_c]
            probability *= self.categorical_probability(x[d], theta_dc)

        return probability
