import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_FILE = Path("team_stats.csv")   # your per-team stats
OUTPUT_FILE = Path("all_matchups.csv")
RANDOM_SEED = 42

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# Expecting:
# TeamName, TeamSlug, Seed, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF, TeamScore, OppScore, WinPct

STAT_COLS = [
    "FGM","FGA","3PM","3PA","FTM","FTA",
    "OR","DR","Ast","TO","Stl","Blk","PF",
    "TeamScore","OppScore","WinPct"
]

COL_MAP = {
    "FGM": "FGM",
    "FGA": "FGA",
    "3PM": "FGM3",
    "3PA": "FGA3",
    "FTM": "FTM",
    "FTA": "FTA",
    "OR": "OR",
    "DR": "DR",
    "Ast": "Ast",
    "TO": "TO",
    "Stl": "Stl",
    "Blk": "Blk",
    "PF": "PF",
    "TeamScore": "TeamScore",
    "OppScore": "OppScore",
    "WinPct": "Win"
}

# =========================
# CROSS JOIN (ALL MATCHUPS)
# =========================
matchups = df.merge(
    df,
    how="cross",
    suffixes=("_T1", "_T2")
)

# Remove self matchups
matchups = matchups[matchups["TeamName_T1"] != matchups["TeamName_T2"]]

# Optional: remove duplicate pairs (A vs B and B vs A)
matchups = matchups[matchups["TeamName_T1"] < matchups["TeamName_T2"]]

# =========================
# RANDOMIZE TEAM ORDER (LIKE TRAINING)
# =========================
rng = np.random.default_rng(RANDOM_SEED)
swap = rng.random(len(matchups)) < 0.5

# =========================
# BUILD FEATURES
# =========================
final = pd.DataFrame({
    "TeamName_T1": np.where(swap, matchups["TeamName_T2"], matchups["TeamName_T1"]),
    "TeamName_T2": np.where(swap, matchups["TeamName_T1"], matchups["TeamName_T2"]),
})

final["T1_Seed"] = np.where(swap, matchups["Seed_T2"], matchups["Seed_T1"])
final["T2_Seed"] = np.where(swap, matchups["Seed_T1"], matchups["Seed_T2"])

final["T1_TeamSlug"] = np.where(swap, matchups["TeamSlug_T2"], matchups["TeamSlug_T1"])
final["T2_TeamSlug"] = np.where(swap, matchups["TeamSlug_T1"], matchups["TeamSlug_T2"])

for input_col, output_col in COL_MAP.items():
    t1 = np.where(swap, matchups[f"{input_col}_T2"], matchups[f"{input_col}_T1"])
    t2 = np.where(swap, matchups[f"{input_col}_T1"], matchups[f"{input_col}_T2"])

    final[f"T1_{output_col}"] = t1
    final[f"T2_{output_col}"] = t2
    final[f"Diff_{output_col}"] = t1 - t2

# =========================
# SAVE
# =========================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(final):,}")
print(f"Columns: {len(final.columns)}")