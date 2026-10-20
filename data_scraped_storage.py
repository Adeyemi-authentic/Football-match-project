"""Fetch finished matches from football-data.org, build pre-match features and store them."""
import os
import sys
import time

import pandas as pd
import requests

from config import API_KEYS, COMBINED_CSV, TABLE_NAME, get_sqlalchemy_engine
from features import COMPETITIONS, build_features

BASE_URL = "https://api.football-data.org/v4/competitions/{code}/matches"

# Optional comma-separated season start years, e.g. SEASONS=2023,2024.
# Empty means the API's current season.
SEASONS = [s.strip() for s in os.getenv("SEASONS", "").split(",") if s.strip()] or [None]


def fetch_matches(code, season=None):
    headers = {"X-Auth-Token": API_KEYS["API_KEY"]}
    params = {"status": "FINISHED"}
    if season:
        params["season"] = season

    for _ in range(3):
        response = requests.get(BASE_URL.format(code=code), headers=headers, params=params, timeout=30)
        if response.status_code == 429:  # free tier allows 10 requests/minute
            print("Rate limited, waiting 60s...")
            time.sleep(60)
            continue
        if response.status_code != 200:
            sys.exit(f"Error fetching {code} (season {season or 'current'}): "
                     f"{response.status_code}, {response.text}")
        break
    else:
        sys.exit(f"Error fetching {code}: still rate limited after retries")

    rows = []
    for match in response.json()["matches"]:
        if match["status"] != "FINISHED":
            continue
        rows.append({
            "id": match["id"],
            "competition": code,
            "season": match["season"]["startDate"][:4],
            "matchday": match["matchday"],
            "date": pd.to_datetime(match["utcDate"]),
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "home_score": match["score"]["fullTime"]["home"],
            "away_score": match["score"]["fullTime"]["away"],
        })
    return rows


def main():
    if not API_KEYS["API_KEY"]:
        sys.exit("API_KEY is not set. Add it to your .env file.")

    rows = []
    for code, name in COMPETITIONS.items():
        for season in SEASONS:
            league_rows = fetch_matches(code, season)
            print(f"{name} ({season or 'current'}): {len(league_rows)} finished matches")
            rows.extend(league_rows)

    matches = pd.DataFrame(rows).drop_duplicates(subset="id")
    data, _ = build_features(matches)

    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(COMBINED_CSV, index=False)
    print(f"Saved {len(data)} matches to {COMBINED_CSV}")

    if os.getenv("GITHUB_ACTIONS") == "true":
        print("Running in GitHub Actions — skipping database upload.")
    else:
        engine = get_sqlalchemy_engine()
        data.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
        print(f"Data uploaded to database table '{TABLE_NAME}'.")


if __name__ == "__main__":
    main()
