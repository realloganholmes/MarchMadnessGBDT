import numpy as np

def split_loss(left_arr, right_arr):
  left_size = len(left_arr)
  right_size = len(right_arr)

  if left_size == 0 or right_size == 0:
    return np.inf

  left_var = np.var(left_arr)
  right_var = np.var(right_arr)

  loss = left_var * left_size + right_var * right_size
  return loss

def best_loss(X, y):

    best_threshold = None
    best_feature = None
    best_loss_val = np.inf

    num_features = X.shape[1]

    for feature_index in range(num_features):

        arr = X[:, feature_index]

        thresholds = np.quantile(arr, np.linspace(0.1,0.9,10))

        for threshold in thresholds:

            mask = arr <= threshold

            left_y = y[mask]
            right_y = y[~mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            loss = (
                np.var(left_y) * len(left_y) +
                np.var(right_y) * len(right_y)
            )

            if loss < best_loss_val:
                best_loss_val = loss
                best_feature = feature_index
                best_threshold = threshold

    return best_threshold, best_feature, best_loss_val