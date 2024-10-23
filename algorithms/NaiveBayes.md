# Naive Bayes Mathematics

Bayes' Theorem is give below:


\[
P(C | X) = \frac{P(X | C) \cdot P(C)}{P(X)}
\]

## Naive Assumption

Naive Bayes assumes that the features are conditionally independent given the class label. 

\[
P(X | C) = P(X_1, X_2, \ldots, X_n | C) = P(X_1 | C) \cdot P(X_2 | C) \cdots P(X_n | C)
\]


The classification rule for Naive Bayes is to choose the class \(C\) that maximizes the posterior probability

\[
\hat{C} = \arg\max_{C} P(C | X) = \arg\max_{C} P(X | C) \cdot P(C)
\]


1. **Training Phase**:
   - Calculate the prior probabilities \(P(C)\) for each class:
     \[
     P(C) = \frac{\text{Number of samples in class } C}{\text{Total number of samples}}
     \]
   - Calculate the likelihood \(P(X_i | C)\) for each feature:
     - For categorical features, use:
     \[
     P(X_i = v | C) = \frac{\text{Count of } v \text{ in class } C}{\text{Total count in class } C}
     \]
     - For continuous features, often assume a normal distribution:
     \[
     P(X_i | C) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_i - \mu)^2}{2\sigma^2}\right)
     \]
     where \(\mu\) is the mean and \(\sigma^2\) is the variance of feature \(X_i\) in class \(C\).

2. **Prediction Phase**:
   - For a new sample \(X\), calculate the posterior probability for each class and choose the one with the highest probability.

