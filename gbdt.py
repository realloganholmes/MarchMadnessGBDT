import numpy as np
from tree import DecisionTree

class GradientBoostedTree:
  def __init__(self, n_estimators, learning_rate, max_depth):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.trees = []

  def fit(self, X, y):
    self.base_value = np.mean(y)
    predictions = np.full(y.shape, np.mean(y))
    residuals = y - predictions

    for i in range(self.n_estimators):
        tree = DecisionTree(max_depth=self.max_depth, min_samples=10)
        tree.fit(X, residuals)
        self.trees.append(tree)
        update = np.array(tree.predict(X)) * self.learning_rate
        predictions += update
        residuals = y - predictions  

  def predict(self, X):
    predictions = np.full(X.shape[0], self.base_value)

    for tree in self.trees:
      predictions += self.learning_rate * tree.predict(X)
    
    return np.clip(predictions, 0, 1)