"""FastAPI backend. Run with: uvicorn app:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import Predictor

predictor = Predictor()

app = FastAPI(title="Football Predictor API", version="2.0.0")


class Match(BaseModel):
    home_team: str
    away_team: str


@app.get("/")
def home():
    return {"message": "Football Predictor API is running", "status": "healthy",
            "data_as_of": predictor.data_as_of}


@app.get("/teams")
def teams():
    """Teams in each competition's latest season, by competition code."""
    return predictor.teams_by_competition()


@app.post("/predict")
def predict(match: Match):
    try:
        result = predictor.predict_match(match.home_team, match.away_team)
        result["features"] = predictor.match_features(match.home_team, match.away_team)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
