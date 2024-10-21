# Naive Bayes Mathematics

Naive Bayes is a family of probabilistic algorithms based on Bayes' Theorem, particularly useful for classification tasks. The key concepts involve understanding conditional probabilities and the assumption of feature independence. Here's a breakdown of the mathematics behind Naive Bayes:

## Bayes' Theorem

Bayes' Theorem provides a way to update the probability estimate for a hypothesis as more evidence becomes available. It is expressed as:

\[
P(C | X) = \frac{P(X | C) \cdot P(C)}{P(X)}
\]

Where:
- \(P(C | X)\) is the posterior probability of class \(C\) given feature \(X\).
- \(P(X | C)\) is the likelihood of feature \(X\) given class \(C\).
- \(P(C)\) is the prior probability of class \(C\).
- \(P(X)\) is the marginal likelihood of feature \(X\).

## Naive Assumption

Naive Bayes assumes that the features are conditionally independent given the class label. This simplifies the calculation of the likelihood \(P(X | C)\) when \(X\) consists of multiple features:

\[
P(X | C) = P(X_1, X_2, \ldots, X_n | C) = P(X_1 | C) \cdot P(X_2 | C) \cdots P(X_n | C)
\]

## Putting It All Together

The overall classification rule for Naive Bayes is to choose the class \(C\) that maximizes the posterior probability:

\[
\hat{C} = \arg\max_{C} P(C | X) = \arg\max_{C} P(X | C) \cdot P(C)
\]

## Steps in Naive Bayes Classifier

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

## Example

Consider a simple binary classification problem with two features:

- **Features**: \(X_1\) (e.g., weather: sunny/rainy), \(X_2\) (e.g., temperature: hot/cold).
- **Classes**: \(C\) (e.g., play or don't play).

1. Calculate \(P(C)\) for "Play" and "Don't Play."
2. Calculate \(P(X_1 | C)\) and \(P(X_2 | C)\) based on the training data.
3. For a new sample, calculate \(P(X | C)\) and select the class with the highest posterior probability.
