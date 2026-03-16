import pandas as pd
import numpy as np
from gbdt import GradientBoostedTree
from sklearn.metrics import log_loss


def run_experiment(config):

    seed = config["seed"]
    np.random.seed(seed)

    df = pd.read_csv("mens_cbb_matchups_rolling.csv")

    eps = 1e-9

    # -------------------------
    # Feature engineering
    # -------------------------

    df["T1_eFG"] = (df["T1_FGM"] + 0.5 * df["T1_FGM3"]) / (df["T1_FGA"] + eps)
    df["T2_eFG"] = (df["T2_FGM"] + 0.5 * df["T2_FGM3"]) / (df["T2_FGA"] + eps)

    df["T1_TO_rate"] = df["T1_TO"] / (df["T1_FGA"] + eps)
    df["T2_TO_rate"] = df["T2_TO"] / (df["T2_FGA"] + eps)

    df["T1_FT_rate"] = df["T1_FTA"] / (df["T1_FGA"] + eps)
    df["T2_FT_rate"] = df["T2_FTA"] / (df["T2_FGA"] + eps)

    df["Diff_eFG"] = df["T1_eFG"] - df["T2_eFG"]
    df["Diff_TO_rate"] = df["T1_TO_rate"] - df["T2_TO_rate"]

    df["T1_REB"] = df["T1_OR"] + df["T1_DR"]
    df["T2_REB"] = df["T2_OR"] + df["T2_DR"]

    df["Diff_REB"] = df["T1_REB"] - df["T2_REB"]

    df["Diff_FT_rate"] = df["T1_FT_rate"] - df["T2_FT_rate"]

    df["T1_poss"] = df["T1_FGA"] - df["T1_OR"] + df["T1_TO"] + 0.475 * df["T1_FTA"]
    df["T2_poss"] = df["T2_FGA"] - df["T2_OR"] + df["T2_TO"] + 0.475 * df["T2_FTA"]

    df["T1_eFG"] = (df["T1_FGM"] + 0.5 * df["T1_FGM3"]) / (df["T1_FGA"] + eps)
    df["T2_eFG"] = (df["T2_FGM"] + 0.5 * df["T2_FGM3"]) / (df["T2_FGA"] + eps)

    df["Diff_eFG"] = df["T1_eFG"] - df["T2_eFG"]

    # -------------------------
    # Feature families
    # -------------------------

    feature_families = {

        "shooting": [
            "Diff_FGM",
            "Diff_FGA",
            "Diff_FGM3",
            "Diff_FGA3",
            "Diff_eFG",
        ],

        "free_throws": [
            "Diff_FTM",
            "Diff_FTA",
            "Diff_FT_rate",
        ],

        "rebounding": [
            "Diff_OR",
            "Diff_DR",
            "Diff_REB",
        ],

        "turnovers": [
            "Diff_TO",
            "Diff_TO_rate",
        ],

        "defense": [
            "Diff_Stl",
            "Diff_Blk",
            "Diff_PF",
        ],

        "playmaking": [
            "Diff_Ast",
        ],

        "win_history": [
            "Diff_Win",
        ]
    }

    # -------------------------
    # Build feature list
    # -------------------------

    feature_cols = []

    for family, cols in feature_families.items():
        if config.get(f"use_{family}", False):
            feature_cols.extend(cols)

    # -------------------------
    # Remove leakage
    # -------------------------

    leakage = [
        "T1_TeamScore",
        "T2_TeamScore",
        "Diff_TeamScore",
        "T1_OppScore",
        "T2_OppScore",
        "Diff_OppScore",
        "T1_Win",
        "T2_Win",
    ]

    feature_cols = [f for f in feature_cols if f not in leakage]

    # -------------------------
    # Train / Test split
    # -------------------------

    train_df = df[df["Season"] < 2010]
    test_df = df[df["Season"] >= 2010]

    X_train = train_df[feature_cols].values
    y_train = train_df["Team1Win"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["Team1Win"].values

    # -------------------------
    # Random training sample
    # -------------------------

    sample_size = config["sample_size"]

    idx = np.random.choice(len(X_train), sample_size, replace=False)

    X_train = X_train[idx]
    y_train = y_train[idx]

    # -------------------------
    # Train model
    # -------------------------

    gbt = GradientBoostedTree(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
    )

    gbt.fit(X_train, y_train)

    predictions = gbt.predict(X_test)

    score = log_loss(y_test, predictions)

    return {
        "log_loss": score
    }


if __name__ == "__main__":

    config = {

        "seed": 42,

        # feature families
        "use_shooting": True,
        "use_free_throws": True,
        "use_rebounding": True,
        "use_turnovers": True,
        "use_defense": True,
        "use_playmaking": True,
        "use_win_history": True,

        # model params
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,

        # dataset size
        "sample_size": 15000
    }

    print(run_experiment(config))