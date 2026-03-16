import pandas as pd
import numpy as np
from gbdt import GradientBoostedTree
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

np.random.seed(42)

df = pd.read_csv("mens_cbb_matchups_rolling.csv")

feature_cols = [col for col in df.columns if col.startswith("Diff_")]

train_df = df[df["Season"] < 2010]
test_df  = df[df["Season"] >= 2010]

X_train = train_df[feature_cols].values
y_train = train_df["Team1Win"].values

X_test = test_df[feature_cols].values
y_test = test_df["Team1Win"].values


# random sampling
sample_size = 10000
indices = np.random.choice(len(X_train), sample_size, replace=False)

X_train_small = X_train[indices]
y_train_small = y_train[indices]


gbt = GradientBoostedTree(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=3
)

gbt.fit(X_train_small, y_train_small)

predictions = gbt.predict(X_test)

pred_class = (predictions > 0.5).astype(int)

mse = mean_squared_error(y_test, predictions)
acc = accuracy_score(y_test, pred_class)
ll = log_loss(y_test, predictions)

print("Log Loss:", ll)
print("MSE:", mse)
print("Accuracy:", acc)
print("Predictions:", predictions[:10])
print("Predicted classes:", pred_class[:10])