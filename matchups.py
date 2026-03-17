# compute_matchups.py
import pandas as pd
from itertools import product

# Load team stats
df = pd.read_csv("team_stats.csv")

# Convert to dict for fast lookup
team_stats_dict = {row["TeamName"]: row.to_dict() for _, row in df.iterrows()}

def matchup_diff(team1, team2):
    t1 = team_stats_dict[team1]
    t2 = team_stats_dict[team2]
    
    # Differences for numeric stats only
    numeric_keys = [k for k in t1.keys() if k not in ("TeamName", "TeamSlug")]
    diff = {f"Diff_{k}": t1[k] - t2[k] for k in numeric_keys}
    
    # Combine all stats
    combined = {f"T1_{k}": t1[k] for k in numeric_keys + ["TeamName"]}
    combined.update({f"T2_{k}": t2[k] for k in numeric_keys + ["TeamName"]})
    combined.update(diff)
    
    # Add team names as simple columns
    combined["Team1"] = team1
    combined["Team2"] = team2
    return combined

# Generate all N^2 matchups (skip self vs self)
matchups = []
teams = list(team_stats_dict.keys())
for t1, t2 in product(teams, teams):
    if t1 != t2:
        matchups.append(matchup_diff(t1, t2))

# Save to CSV
df_matchups = pd.DataFrame(matchups)
df_matchups.to_csv("all_matchups.csv", index=False)
print("Saved all matchup stats to all_matchups.csv")