import streamlit as st
import pandas as pd

from features import COMPETITIONS
from model import MODEL_PATH, Predictor

# Set page config
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

OUTCOMES = ["Home Win", "Draw", "Away Win"]

FORM_STATS = [
    ("elo", "Elo rating", "Overall strength from all results so far. 1500 is average."),
    ("ppg", "Points per game (season)", "3 for a win, 1 for a draw."),
    ("gf_pg", "Goals scored per game", ""),
    ("ga_pg", "Goals conceded per game", ""),
    ("venue_ppg", "Points per game (home/away)", "Home games for the home team, away games for the away team."),
    ("form_ppg", "Points per game (last 5)", ""),
    ("form_gd", "Goal difference per game (last 5)", ""),
]


# Load the model
@st.cache_resource
def load_predictor():
    try:
        return Predictor(MODEL_PATH), None
    except FileNotFoundError:
        return None, f"Model file '{MODEL_PATH}' not found. Run `python train.py` first."
    except ValueError as e:
        return None, str(e)

predictor, load_error = load_predictor()


def show_probabilities(home_team, away_team, probabilities):
    most_likely = max(probabilities, key=probabilities.get)
    labels = {"Home Win": f"{home_team} win", "Draw": "Draw", "Away Win": f"{away_team} win"}

    cols = st.columns(3)
    for col, outcome in zip(cols, OUTCOMES):
        with col, st.container(border=True):
            st.metric(labels[outcome], f"{probabilities[outcome]:.0%}")
            st.progress(probabilities[outcome])
            st.caption("Most likely" if outcome == most_likely else " ")

    st.caption(
        "These are probabilities, not a sure pick: even the most likely result often doesn't happen. "
        "A draw is rarely the single most likely result, but it happens in about a quarter of matches."
    )


def main():
    st.title("⚽ Football Match Predictor")
    st.write("Win, draw and loss probabilities for matches in seven European leagues, "
             "based on each team's results so far.")

    if predictor is None:
        st.error(load_error)
        st.stop()

    # Sidebar for information
    with st.sidebar:
        st.header("About")
        st.write(
            "Proudly designed by Osinowo Abdulazeez. Pick a league and two teams to see how likely "
            "each result is, based on each team's current strength and form."
        )

        st.header("How good is it?")
        lr = predictor.metrics.get("logistic_regression")
        home = predictor.metrics.get("always_home_win")
        if lr and home:
            st.write(f"On recent matches held back for testing, its most likely result was right "
                     f"**{lr['accuracy']:.1%}** of the time. Always picking the home team was right "
                     f"{home['accuracy']:.1%} of the time.")
            st.caption("Football is hard to predict: bookmakers typically manage around 50–53%.")

        st.header("Model")
        st.write(f"Logistic regression on {predictor.input_size} features (C={predictor.C})")
        st.write(f"Results up to **{predictor.data_as_of}**")

    # Match selection
    teams_by_competition = predictor.teams_by_competition()
    competition = st.selectbox("League", list(teams_by_competition),
                               format_func=lambda code: COMPETITIONS.get(code, code))
    teams = teams_by_competition[competition]

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home team", teams, index=0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("Away team", away_options, index=0)

    # Predictions are instant, so they update as soon as a team changes
    result = predictor.predict_match(home_team, away_team)
    st.subheader(f"{home_team} vs {away_team}")
    show_probabilities(home_team, away_team, result["probabilities"])

    # Current form of both teams
    features = predictor.match_features(home_team, away_team)
    stats = pd.DataFrame({
        "Stat": [label for _, label, _ in FORM_STATS],
        home_team: [features[f"home_{key}"] for key, _, _ in FORM_STATS],
        away_team: [features[f"away_{key}"] for key, _, _ in FORM_STATS],
        "Notes": [note for _, _, note in FORM_STATS],
    })
    with st.expander("Team form"):
        st.dataframe(stats.style.format({home_team: "{:.2f}", away_team: "{:.2f}"}),
                     width="stretch", hide_index=True)
        st.caption("Per-game rates are smoothed towards league averages for teams with few games played. "
                   "The Elo rating matters most to the model.")

    # Footer
    st.divider()
    st.caption("Made using Streamlit and scikit-learn. For educational and entertainment purposes.")

if __name__ == "__main__":
    main()
