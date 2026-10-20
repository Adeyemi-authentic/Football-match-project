"""Feature engineering shared by the scraper, the training script and the apps.

Matches are replayed in date order while a running state is kept for every
team. Each match's features are read from the two teams' states *before* the
match is applied, so they never include the result being predicted. The final
states are saved with the model so the apps can predict upcoming matches.
"""
import numpy as np
import pandas as pd

COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "DED": "Eredivisie",
    "FL1": "Ligue 1",
    "PPL": "Primeira Liga",
}

LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}

# Elo ratings
ELO_START = 1500.0
ELO_K = 15.0
ELO_HOME_ADVANTAGE = 60.0
ELO_SEASON_CARRY = 1.0  # share of a rating's distance from 1500 kept into a new season
ELO_PROMOTED = -50.0    # starting offset for teams that first appear after the first season (promoted)

# Per-game rates are shrunk towards typical league values so teams with only a
# few games played don't get extreme numbers: (total + prior * SHRINK) / (games + SHRINK)
SHRINK = 3
FORM_GAMES = 5
PRIOR_PPG = 1.37
PRIOR_GOALS = 1.4
PRIOR_VENUE_PPG = {"home": 1.6, "away": 1.15}

_SIDE_FEATURES = ["elo", "ppg", "gf_pg", "ga_pg", "venue_ppg", "form_ppg", "form_gd"]
FEATURE_COLUMNS = [f"{side}_{name}" for name in _SIDE_FEATURES for side in ("home", "away")]


def new_team_state(team, competition, season, elo=ELO_START):
    return {
        "team": team,
        "competition": competition,
        "season": season,
        "elo": elo,
        "played": 0, "points": 0, "gf": 0, "ga": 0,
        "venue": {"home": [0, 0], "away": [0, 0]},  # [played, points]
        "recent": [],  # [points, goal difference] of the last FORM_GAMES matches
    }


def _start_season(state, competition, season):
    if state["season"] == season:
        return
    state.update(competition=competition, season=season, played=0, points=0, gf=0, ga=0,
                 venue={"home": [0, 0], "away": [0, 0]})
    state["elo"] = ELO_START + (state["elo"] - ELO_START) * ELO_SEASON_CARRY


def _rate(total, games, prior):
    return (total + prior * SHRINK) / (games + SHRINK)


def _side_features(state, venue):
    venue_played, venue_points = state["venue"][venue]
    recent = state["recent"]
    return {
        "elo": state["elo"],
        "ppg": _rate(state["points"], state["played"], PRIOR_PPG),
        "gf_pg": _rate(state["gf"], state["played"], PRIOR_GOALS),
        "ga_pg": _rate(state["ga"], state["played"], PRIOR_GOALS),
        "venue_ppg": _rate(venue_points, venue_played, PRIOR_VENUE_PPG[venue]),
        "form_ppg": _rate(sum(p for p, _ in recent), len(recent), PRIOR_PPG),
        "form_gd": _rate(sum(gd for _, gd in recent), len(recent), 0.0),
    }


def match_features(home_state, away_state):
    """Feature dict for a match between two teams in their current states."""
    home = _side_features(home_state, "home")
    away = _side_features(away_state, "away")
    features = {}
    for name in _SIDE_FEATURES:
        features[f"home_{name}"] = float(home[name])
        features[f"away_{name}"] = float(away[name])
    return features


def _points(gf, ga):
    return 3 if gf > ga else 1 if gf == ga else 0


def _apply_result(state, venue, gf, ga):
    points = _points(gf, ga)
    state["played"] += 1
    state["points"] += points
    state["gf"] += gf
    state["ga"] += ga
    state["venue"][venue][0] += 1
    state["venue"][venue][1] += points
    state["recent"] = (state["recent"] + [[points, gf - ga]])[-FORM_GAMES:]


def _apply_elo(home, away, home_score, away_score):
    expected_home = 1 / (1 + 10 ** ((away["elo"] - home["elo"] - ELO_HOME_ADVANTAGE) / 400))
    actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
    margin = abs(home_score - away_score)
    multiplier = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
    change = ELO_K * multiplier * (actual_home - expected_home)
    home["elo"] += change
    away["elo"] -= change


def build_features(matches: pd.DataFrame):
    """Return (matches with Result/GG/feature columns, final team states).

    Expects columns: id, competition, season, date, home_team, away_team,
    home_score, away_score.
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["season"] = df["season"].astype(str)
    df = df.sort_values(["date", "id"]).reset_index(drop=True)

    first_season = df["season"].min()
    states = {}
    rows = []
    for m in df.itertuples(index=False):
        for team in (m.home_team, m.away_team):
            if team not in states:
                elo = ELO_START + (ELO_PROMOTED if m.season != first_season else 0.0)
                states[team] = new_team_state(team, m.competition, m.season, elo)
            state = states[team]
            _start_season(state, m.competition, m.season)
        home, away = states[m.home_team], states[m.away_team]

        rows.append(match_features(home, away))

        home_score, away_score = int(m.home_score), int(m.away_score)
        _apply_elo(home, away, home_score, away_score)
        _apply_result(home, "home", home_score, away_score)
        _apply_result(away, "away", away_score, home_score)

    df = pd.concat([df, pd.DataFrame(rows, columns=FEATURE_COLUMNS)], axis=1)

    # 0 = Away Win, 1 = Draw, 2 = Home Win
    df["Result"] = np.select([df["home_score"] < df["away_score"],
                              df["home_score"] == df["away_score"]], [0, 1], 2)
    df["GG"] = ((df["home_score"] >= 1) & (df["away_score"] >= 1)).astype(int)
    return df, states
