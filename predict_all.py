import pandas as pd
import numpy as np
from gbdt import GradientBoostedTree
import pickle

# =========================
# CONFIG
# =========================
MATCHUP_FILE = "all_matchups.csv"
MODEL_FILE = "gbdt_model.pkl"
OUTPUT_FILE = "matchup_predictions.csv"

# same config as training
config = {
    "use_shooting": True,
    "use_free_throws": True,
    "use_rebounding": True,
    "use_turnovers": True,
    "use_defense": True,
    "use_playmaking": True,
    "use_win_history": True,
}

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(MATCHUP_FILE)

eps = 1e-9

# =========================
# FEATURE ENGINEERING (IDENTICAL)
# =========================

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

# =========================
# FEATURE FAMILIES (IDENTICAL)
# =========================

feature_families = {
    "shooting": [
        "Diff_FGM","Diff_FGA","Diff_FGM3","Diff_FGA3","Diff_eFG",
    ],
    "free_throws": [
        "Diff_FTM","Diff_FTA","Diff_FT_rate",
    ],
    "rebounding": [
        "Diff_OR","Diff_DR","Diff_REB",
    ],
    "turnovers": [
        "Diff_TO","Diff_TO_rate",
    ],
    "defense": [
        "Diff_Stl","Diff_Blk","Diff_PF",
    ],
    "playmaking": [
        "Diff_Ast",
    ],
    "win_history": [
        "Diff_Win",
    ]
}

# =========================
# BUILD FEATURE LIST
# =========================

feature_cols = []
for family, cols in feature_families.items():
    if config.get(f"use_{family}", False):
        feature_cols.extend(cols)

# remove leakage (same as training)
leakage = [
    "T1_TeamScore","T2_TeamScore","Diff_TeamScore",
    "T1_OppScore","T2_OppScore","Diff_OppScore",
    "T1_Win","T2_Win",
]

feature_cols = [f for f in feature_cols if f not in leakage]

# =========================
# LOAD MODEL
# =========================

with open(MODEL_FILE, "rb") as f:
    gbt = pickle.load(f)

# =========================
# PREDICT
# =========================

X = df[feature_cols].values
probs = gbt.predict(X)

df["T1_Win_Prob"] = probs
df["T2_Win_Prob"] = 1 - probs

# =========================
# SAVE
# =========================

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved predictions → {OUTPUT_FILE}")
print(df[["TeamName_T1","TeamName_T2","T1_Win_Prob"]].head())