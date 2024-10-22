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
