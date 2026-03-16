import numpy as np
from loss import best_loss

class Node:
  def __init__(self, feature=None, threshold=None, right=None, left=None, value=None):
    self.feature = feature
    self.threshold = threshold
    self.right = right
    self.left = left
    self.value = value
  
  def is_leaf(self):
    return self.value is not None
  
  def predict_row(self, row):
    if self.is_leaf():
      return self.value
    
    if row[self.feature] > self.threshold:
      return self.left.predict_row(row)
    else:
      return self.right.predict_row(row)
    
class DecisionTree:
  def __init__(self, max_depth=None, min_samples=None, root=None):
    self.max_depth = max_depth
    self.min_samples = min_samples
    self.root = root

  def fit(self, X, y):
    self.root = self.build_tree(X, y, 0)

  def predict(self, X):
    return np.array([self.root.predict_row(row) for row in X])

  def build_tree(self, X, y, depth=0):
    threshold, feature, loss = best_loss(X, y)

    if depth >= self.max_depth or len(y) < self.min_samples:
        return Node(value=np.mean(y))

    mask = X[:, feature] <= threshold

    left_X = X[mask]
    left_y = y[mask]

    right_X = X[~mask]
    right_y = y[~mask]

    left_child = self.build_tree(left_X, left_y, depth + 1)
    right_child = self.build_tree(right_X, right_y, depth + 1)

    return Node(feature, threshold, right_child, left_child)