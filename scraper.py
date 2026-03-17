# scrape_team_stats.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import os

OUTPUT_FILE = "team_stats.csv"

team_slugs = {
    "Duke": "duke",
    "UConn": "connecticut",
    "Michigan State": "michigan-state",
    "Kansas": "kansas",
    "St. John's": "st-johns-ny",
    "Louisville": "louisville",
    "UCLA": "ucla",
    "Ohio State": "ohio-state",
    "TCU": "texas-christian",
    "UCF": "central-florida",
    "South Florida": "south-florida",
    "Northern Iowa": "northern-iowa",
    "California Baptist": "california-baptist",
    "North Dakota State": "north-dakota-state",
    "Furman": "furman",
    "Siena": "siena",
    "Arizona": "arizona",
    "Purdue": "purdue",
    "Gonzaga": "gonzaga",
    "Arkansas": "arkansas",
    "Wisconsin": "wisconsin",
    "BYU": "brigham-young",
    "Miami (FL)": "miami-fl",
    "Villanova": "villanova",
    "Utah State": "utah-state",
    "Missouri": "missouri",
    "Texas": "texas",
    "High Point": "high-point",
    "Hawaii": "hawaii",
    "Kennesaw State": "kennesaw-state",
    "Queens": "queens-nc",
    "Florida": "florida",
    "Houston": "houston",
    "Illinois": "illinois",
    "Nebraska": "nebraska",
    "Vanderbilt": "vanderbilt",
    "North Carolina": "north-carolina",
    "St. Mary's": "saint-marys-ca",
    "Clemson": "clemson",
    "Iowa": "iowa",
    "Texas A&M": "texas-am",
    "VCU": "virginia-commonwealth",
    "McNeese": "mcneese-state",
    "Troy": "troy",
    "Idaho": "idaho",
    "Prairie View A&M": "prairie-view",
    "Michigan": "michigan",
    "Iowa State": "iowa-state",
    "Virginia": "virginia",
    "Alabama": "alabama",
    "Texas Tech": "texas-tech",
    "Tennessee": "tennessee",
    "Kentucky": "kentucky",
    "Georgia": "georgia",
    "Saint Louis": "saint-louis",
    "Santa Clara": "santa-clara",
    "Miami (OH)": "miami-oh",
    "SMU": "southern-methodist",
    "Akron": "akron",
    "Hofstra": "hofstra",
    "Wright State": "wright-state",
    "Tennessee State": "tennessee-state",
    "UMBC": "maryland-baltimore-county",
    "Howard": "howard",
    "LIU": "long-island-university",
    "Penn": "pennsylvania",
    "NC State": "north-carolina-state",
    "Lehigh": "lehigh"
}

season_year = 2026


def get_team_stats(team_name, team_slug):
    url = f"https://www.sports-reference.com/cbb/schools/{team_slug}/{season_year}-gamelogs.html"

    try:
        res = requests.get(url, timeout=10)
    except Exception as e:
        print(f"Request failed for {team_name}: {e}")
        return None

    if res.status_code != 200:
        print(f"Failed to fetch {team_name}")
        return None

    soup = BeautifulSoup(res.content, "html.parser")
    table = soup.find("table", id="team_game_log")

    if table is None:
        print(f"No gamelog table for {team_name}")
        return None

    df = pd.read_html(str(table))[0]

    df = df[df[('Unnamed: 0_level_0', 'Rk')] != 'Rk']

    numeric_cols = [
        ('Team', 'FG'), ('Team', 'FGA'),
        ('Team', '3P'), ('Team', '3PA'),
        ('Team', 'FT'), ('Team', 'FTA'),
        ('Team', 'ORB'), ('Team', 'DRB'),
        ('Team', 'AST'), ('Team', 'STL'),
        ('Team', 'BLK'), ('Team', 'TOV'),
        ('Team', 'PF'),
        ('Score', 'Tm'),
        ('Score', 'Opp')
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[('Team', 'FG'), ('Team', 'FGA')])

    if len(df) < 10:
        print(f"Not enough games for {team_name}")
        return None

    last10 = df.iloc[:-1].tail(10)
    
    # Compute averages
    stats = {
        "TeamName": team_name,
        "TeamSlug": team_slug,

        "FGM": last10[('Team', 'FG')].mean(),
        "FGA": last10[('Team', 'FGA')].mean(),
        "3PM": last10[('Team', '3P')].mean(),
        "3PA": last10[('Team', '3PA')].mean(),
        "FTM": last10[('Team', 'FT')].mean(),
        "FTA": last10[('Team', 'FTA')].mean(),
        "OR": last10[('Team', 'ORB')].mean(),
        "DR": last10[('Team', 'DRB')].mean(),
        "Ast": last10[('Team', 'AST')].mean(),
        "TO": last10[('Team', 'TOV')].mean(),
        "Stl": last10[('Team', 'STL')].mean(),
        "Blk": last10[('Team', 'BLK')].mean(),
        "PF": last10[('Team', 'PF')].mean(),

        "TeamScore": last10[('Score', 'Tm')].mean(),
        "OppScore": last10[('Score', 'Opp')].mean(),

        # Win %
        "WinPct": (last10[('Score', 'Rslt')] == 'W').mean()
    }

    return stats

# -----------------------------
# Load existing progress
# -----------------------------
scraped_teams = set()

if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    scraped_teams = set(existing_df["TeamName"])
    print(f"Resuming. Already have {len(scraped_teams)} teams.")

# -----------------------------
# Scrape remaining teams
# -----------------------------
for team, slug in team_slugs.items():

    if team in scraped_teams:
        print(f"Skipping {team} (already scraped)")
        continue

    print(f"Fetching {team}")

    stats = get_team_stats(team, slug)

    if stats:
        df = pd.DataFrame([stats])

        # append immediately
        df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not os.path.exists(OUTPUT_FILE),
            index=False
        )

        print(f"Saved {team}")

    sleep(4)

print("Scraping complete.")