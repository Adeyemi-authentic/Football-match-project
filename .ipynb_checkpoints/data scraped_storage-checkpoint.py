import pandas as pd
import requests
import numpy as np
API_KEY = "53267ff6599941f39feca4342aadee5f"
url = " http://api.football-data.org/v4/competitions/PPL/matches"
headers = {
    "X-Auth-Token": API_KEY
}
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']
id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

ppl_df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date

})

ppl_df["date"] = pd.to_datetime(ppl_df["date"])
ppl_df["date"] = ppl_df["date"].dt.date

ppl_df['home_points'] = ppl_df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

ppl_df['away_points'] = ppl_df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

ppl_df['Home Total Goals Scored'] = 0

ppl_df['Home Total Goals Scored'] = 0
ppl_df['Away Total Goals Scored'] = 0
ppl_df['Home Goals Conceded'] = 0
ppl_df['Away Goals Conceded'] = 0
ppl_df['Home Total Points'] = 0
ppl_df['Away Total Points'] = 0

teams = ppl_df['home_team'].unique()

for team in teams:
    home_goals_scored = ppl_df.loc[ppl_df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = ppl_df.loc[ppl_df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = ppl_df.loc[ppl_df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = ppl_df.loc[ppl_df['away_team'] == team, 'home_score'].cumsum()

    ppl_df.loc[ppl_df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    ppl_df.loc[ppl_df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    ppl_df.loc[ppl_df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    ppl_df.loc[ppl_df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = ppl_df.loc[ppl_df['home_team'] == team].apply(
        lambda row: 3 if row['home_score'] > row['away_score'] else (
            1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = ppl_df.loc[ppl_df['away_team'] == team].apply(
        lambda row: 3 if row['away_score'] > row['home_score'] else (
            1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    ppl_df.loc[ppl_df['home_team'] == team, 'Home Total Points'] = home_points
    ppl_df.loc[ppl_df['away_team'] == team, 'Away Total Points'] = away_points

ppl_df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\portugal league2.csv", index=False)

sa_url = " http://api.football-data.org/v4/competitions/SA/matches"

response = requests.get(sa_url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']
id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

sa_df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date
})

sa_df["date"] = pd.to_datetime(sa_df["date"])
sa_df["date"] = sa_df["date"].dt.date

sa_df['home_points'] = sa_df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

sa_df['away_points'] = sa_df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

sa_df['Home Total Goals Scored'] = 0
sa_df['Away Total Goals Scored'] = 0
sa_df['Home Goals Conceded'] = 0
sa_df['Away Goals Conceded'] = 0
sa_df['Home Total Points'] = 0
sa_df['Away Total Points'] = 0

teams = sa_df['home_team'].unique()

for team in teams:
    home_goals_scored = sa_df.loc[sa_df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = sa_df.loc[sa_df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = sa_df.loc[sa_df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = sa_df.loc[sa_df['away_team'] == team, 'home_score'].cumsum()

    sa_df.loc[sa_df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    sa_df.loc[sa_df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    sa_df.loc[sa_df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    sa_df.loc[sa_df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = sa_df.loc[sa_df['home_team'] == team].apply(
        lambda row: 3 if row['home_score'] > row['away_score'] else (
            1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = sa_df.loc[sa_df['away_team'] == team].apply(
        lambda row: 3 if row['away_score'] > row['home_score'] else (
            1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    sa_df.loc[sa_df['home_team'] == team, 'Home Total Points'] = home_points
    sa_df.loc[sa_df['away_team'] == team, 'Away Total Points'] = away_points

sa_df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\seria A.csv", index=False)

API_KEY = "53267ff6599941f39feca4342aadee5f"
b_url = " http://api.football-data.org/v4/competitions/BL1/matches"

headers = {
    "X-Auth-Token": API_KEY
}

response = requests.get(b_url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']
id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

b_df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date
})

b_df["date"] = pd.to_datetime(b_df["date"])
b_df["date"] = b_df["date"].dt.date

b_df['home_points'] = b_df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

b_df['away_points'] = b_df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

b_df['Home Total Goals Scored'] = 0
b_df['Away Total Goals Scored'] = 0
b_df['Home Goals Conceded'] = 0
b_df['Away Goals Conceded'] = 0
b_df['Home Total Points'] = 0
b_df['Away Total Points'] = 0

teams = b_df['home_team'].unique()

for team in teams:
    home_goals_scored = b_df.loc[b_df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = b_df.loc[b_df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = b_df.loc[b_df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = b_df.loc[b_df['away_team'] == team, 'home_score'].cumsum()

    b_df.loc[b_df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    b_df.loc[b_df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    b_df.loc[b_df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    b_df.loc[b_df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = b_df.loc[b_df['home_team'] == team].apply(
        lambda row: 3 if row['home_score'] > row['away_score'] else (
            1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = b_df.loc[b_df['away_team'] == team].apply(
        lambda row: 3 if row['away_score'] > row['home_score'] else (
            1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    b_df.loc[b_df['home_team'] == team, 'Home Total Points'] = home_points
    b_df.loc[b_df['away_team'] == team, 'Away Total Points'] = away_points

b_df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\bundesliga data.csv", index=False)

API_KEY = "53267ff6599941f39feca4342aadee5f"
f_url = " http://api.football-data.org/v4/competitions/FL1/matches"

headers = {
    "X-Auth-Token": API_KEY
}
response = requests.get(f_url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']
id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

f_df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date
})

f_df["date"] = pd.to_datetime(f_df["date"])
f_df["date"] = f_df["date"].dt.date

f_df['home_points'] = f_df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

f_df['away_points'] = f_df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

f_df['Home Total Goals Scored'] = 0
f_df['Away Total Goals Scored'] = 0
f_df['Home Goals Conceded'] = 0
f_df['Away Goals Conceded'] = 0
f_df['Home Total Points'] = 0
f_df['Away Total Points'] = 0

teams = f_df['home_team'].unique()

for team in teams:
    home_goals_scored = f_df.loc[f_df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = f_df.loc[f_df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = f_df.loc[f_df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = f_df.loc[f_df['away_team'] == team, 'home_score'].cumsum()

    f_df.loc[f_df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    f_df.loc[f_df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    f_df.loc[f_df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    f_df.loc[f_df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = f_df.loc[f_df['home_team'] == team].apply(
        lambda row: 3 if row['home_score'] > row['away_score'] else (
            1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = f_df.loc[f_df['away_team'] == team].apply(
        lambda row: 3 if row['away_score'] > row['home_score'] else (
            1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    f_df.loc[f_df['home_team'] == team, 'Home Total Points'] = home_points
    f_df.loc[f_df['away_team'] == team, 'Away Total Points'] = away_points

f_df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\ligue1.csv", index=False)

API_KEY = "53267ff6599941f39feca4342aadee5f"
laliga_url = " http://api.football-data.org/v4/competitions/PD/matches"

headers = {
    "X-Auth-Token": API_KEY
}

response = requests.get(laliga_url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']
id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

laliga_df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date
})

laliga_df["date"] = pd.to_datetime(laliga_df["date"])
laliga_df["date"] = laliga_df["date"].dt.date

laliga_df['home_points'] = laliga_df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

laliga_df['away_points'] = laliga_df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

laliga_df['Home Total Goals Scored'] = 0
laliga_df['Away Total Goals Scored'] = 0
laliga_df['Home Goals Conceded'] = 0
laliga_df['Away Goals Conceded'] = 0
laliga_df['Home Total Points'] = 0
laliga_df['Away Total Points'] = 0

teams = laliga_df['home_team'].unique()

for team in teams:
    home_goals_scored = laliga_df.loc[laliga_df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = laliga_df.loc[laliga_df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = laliga_df.loc[laliga_df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = laliga_df.loc[laliga_df['away_team'] == team, 'home_score'].cumsum()

    laliga_df.loc[laliga_df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    laliga_df.loc[laliga_df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    laliga_df.loc[laliga_df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    laliga_df.loc[laliga_df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = laliga_df.loc[laliga_df['home_team'] == team].apply(
        lambda row: 3 if row['home_score'] > row['away_score'] else (
            1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = laliga_df.loc[laliga_df['away_team'] == team].apply(
        lambda row: 3 if row['away_score'] > row['home_score'] else (
            1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    laliga_df.loc[laliga_df['home_team'] == team, 'Home Total Points'] = home_points
    laliga_df.loc[laliga_df['away_team'] == team, 'Away Total Points'] = away_points

laliga_df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\laliga.csv", index=False)

API_KEY = "53267ff6599941f39feca4342aadee5f"
ned_url = " http://api.football-data.org/v4/competitions/DED/matches"

headers = {
    "X-Auth-Token": API_KEY
}

response = requests.get(ned_url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']
id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

n_df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date
})

n_df["date"] = pd.to_datetime(n_df["date"])
n_df["date"] = n_df["date"].dt.date

n_df['home_points'] = n_df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

n_df['away_points'] = n_df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

n_df['Home Total Goals Scored'] = 0
n_df['Away Total Goals Scored'] = 0
n_df['Home Goals Conceded'] = 0
n_df['Away Goals Conceded'] = 0
n_df['Home Total Points'] = 0
n_df['Away Total Points'] = 0

teams = n_df['home_team'].unique()

for team in teams:
    home_goals_scored = n_df.loc[n_df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = n_df.loc[n_df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = n_df.loc[n_df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = n_df.loc[n_df['away_team'] == team, 'home_score'].cumsum()

    n_df.loc[n_df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    n_df.loc[n_df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    n_df.loc[n_df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    n_df.loc[n_df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = n_df.loc[n_df['home_team'] == team].apply(
        lambda row: 3 if row['home_score'] > row['away_score'] else (
            1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = n_df.loc[n_df['away_team'] == team].apply(
        lambda row: 3 if row['away_score'] > row['home_score'] else (
            1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    n_df.loc[n_df['home_team'] == team, 'Home Total Points'] = home_points
    n_df.loc[n_df['away_team'] == team, 'Away Total Points'] = away_points

n_df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\eredivise.csv", index=False)

API_KEY = "53267ff6599941f39feca4342aadee5f"
url = " http://api.football-data.org/v4/competitions/PL/matches"
headers = {
    "X-Auth-Token": API_KEY
}

# make api request
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print(f"Error:{response.status_code},{response.json()}")
    exit()

data = response.json()
matches = data['matches']

id = []
home_team = []
away_team = []
home_score = []
away_score = []
Result = []
matchday = []
date = []

for match in matches:
    if match['status'] == "FINISHED":
        id.append(match['id'])
        home_team.append(match['homeTeam']['name'])
        away_team.append(match['awayTeam']['name'])
        home_score.append(match['score']['fullTime']['home'])
        away_score.append(match['score']['fullTime']['away'])
        Result.append(match['score']['winner'])
        matchday.append(match['matchday'])
        date.append(match['utcDate'])

df = pd.DataFrame({
    "id": id,
    "home_team": home_team,
    "away_team": away_team,
    "home_score": home_score,
    "away_score": away_score,
    "Result": Result,
    "matchday": matchday,
    "date": date
})

df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].dt.date

df['home_points'] = df.apply(lambda x: 3 if x['home_score'] > x['away_score'] else 1
if x['home_score'] == x['away_score'] else 0, axis=1)

df['away_points'] = df.apply(lambda x: 3 if x['away_score'] > x['home_score'] else 1
if x['away_score'] == x['home_score'] else 0, axis=1)

df['Home Total Goals Scored'] = 0
df['Away Total Goals Scored'] = 0
df['Home Goals Conceded'] = 0
df['Away Goals Conceded'] = 0
df['Home Total Points'] = 0
df['Away Total Points'] = 0

teams = df['home_team'].unique()

for team in teams:
    home_goals_scored = df.loc[df['home_team'] == team, 'home_score'].cumsum()
    away_goals_scored = df.loc[df['away_team'] == team, 'away_score'].cumsum()
    home_goals_conceded = df.loc[df['home_team'] == team, 'away_score'].cumsum()
    away_goals_conceded = df.loc[df['away_team'] == team, 'home_score'].cumsum()

    df.loc[df['home_team'] == team, 'Home Total Goals Scored'] = home_goals_scored
    df.loc[df['away_team'] == team, 'Away Total Goals Scored'] = away_goals_scored
    df.loc[df['home_team'] == team, 'Home Goals Conceded'] = home_goals_conceded
    df.loc[df['away_team'] == team, 'Away Goals Conceded'] = away_goals_conceded

    home_points = df.loc[df['home_team'] == team].apply(lambda row: 3 if row['home_score'] > row['away_score'] else (
        1 if row['home_score'] == row['away_score'] else 0), axis=1).cumsum()
    away_points = df.loc[df['away_team'] == team].apply(lambda row: 3 if row['away_score'] > row['home_score'] else (
        1 if row['away_score'] == row['home_score'] else 0), axis=1).cumsum()

    df.loc[df['home_team'] == team, 'Home Total Points'] = home_points
    df.loc[df['away_team'] == team, 'Away Total Points'] = away_points

df.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\EPL.csv", index=False)

# cpmbining all csv files together


import os
import glob

os.chdir(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2")

extension = "csv"
all_filename = [i for i in glob.glob('*.{}'.format(extension))]

combined_csv = pd.concat([pd.read_csv(f) for f in all_filename])

combined_csv.duplicated().any()
clean_combined = combined_csv.drop_duplicates()

clean_combined.to_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\clean_combined.csv", index=False)

data = pd.read_csv(r"C:\Users\USER\OneDrive\Documents\Football Api datasets2\clean_combined.csv")

data2 = data.copy()
data2["GG"] = np.where((data2["home_score"] >= 1) & (data2["away_score"] >= 1), "YES", "NO")

# Create a cumulative matches played column for the home team
data2['Cumulative_Home_Matches'] = data2.groupby('home_team').cumcount() + 1

# Create a cumulative matches played column for the away team
data2['Cumulative_Away_Matches'] = data2.groupby('away_team').cumcount() + 1

data2["home_goal_ratio"] = data2["Home Total Goals Scored"] / data2["Cumulative_Home_Matches"]
data2["away_goal_ratio"] = data2["Away Total Goals Scored"] / data2["Cumulative_Away_Matches"]

data2['home_wins'] = (data2['Result'] == 'HOME_TEAM').astype(int)
data2['home_draws'] = (data2['Result'] == 'DRAW').astype(int)
data2['home_losses'] = (data2['Result'] == 'AWAY_TEAM').astype(int)
data2['away_wins'] = (data2['Result'] == 'AWAY_TEAM').astype(int)
data2['away_draws'] = (data2['Result'] == 'DRAW').astype(int)
data2['away_losses'] = (data2['Result'] == 'HOME_TEAM').astype(int)
data2['Cumulative_Home_Wins'] = data2.groupby('home_team')['home_wins'].cumsum()
data2['Cumulative_Home_Draws'] = data2.groupby('home_team')['home_draws'].cumsum()
data2['Cumulative_Home_Losses'] = data2.groupby('home_team')['home_losses'].cumsum()
data2['Cumulative_Away_Wins'] = data2.groupby('away_team')['away_wins'].cumsum()
data2['Cumulative_Away_Draws'] = data2.groupby('away_team')['away_draws'].cumsum()
data2['Cumulative_Away_Losses'] = data2.groupby('away_team')['away_losses'].cumsum()

data2["home_team"] = data2["home_team"].astype("category").cat.codes

data2["away_team"] = data2["away_team"].astype("category").cat.codes
data2["Result"] = data2["Result"].astype("category").cat.codes
data2["GG"] = data2["GG"].astype("category").cat.codes

# pip install pymysql sqlalchemy pandas

from sqlalchemy import create_engine

# using my sql credentials
# Using secure configuration
from config import get_sqlalchemy_engine

engine = get_sqlalchemy_engine()
data2.to_sql("matches_data", schema="FootballML", con=engine, if_exists='replace', index=False)





