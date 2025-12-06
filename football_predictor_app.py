import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import time
import requests
import json

# Set page config
st.set_page_config(
    page_title="⚽ Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00c851, #007e33);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .feature-input {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00c851, #007e33);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 200, 81, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# PyTorch Model Definition
class FootballML(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_size2,output_dim,dropout_prob=0.35):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2=nn.Linear(hidden_size,hidden_size2)
        self.dropout2=nn.Dropout(dropout_prob)
        self.fc3=nn.Linear(hidden_size2,output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply first linear layer and activation
        x = self.dropout1(x)    # Apply dropout1
        x=torch.relu(self.fc2(x))  # second hidden layer activation
        x=self.dropout2(x)
        output = self.fc3(x)         # Final linear layer
        return output

# FastAPI Backend Setup
app = FastAPI(title="Football Predictor API", version="1.0.0")

class MatchFeatures(BaseModel):
    features: List[float]

# Model parameters
input_size=14
hidden_size=42
hidden_size2=18
dropout_prob=0.35
output_dim=3

# Load the model
@st.cache_resource
def load_model():
    try:
        model = FootballML(input_size, hidden_size,hidden_size2,output_dim, dropout_prob)
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'model.pth' not found. Please ensure the model file is in the same directory.")
        return None

model = load_model()

# FastAPI endpoints
@app.get("/")
def home():
    return {"message": "Football Predictor API is running", "status": "healthy"}

@app.post("/predict")
def predict_api(data: MatchFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(data.features) != input_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {input_size} features, got {len(data.features)}"
        )

    inputs = torch.tensor([data.features], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # Get probabilities
    probabilities = torch.softmax(outputs, dim=1)[0]
    predicted_class = torch.argmax(outputs, dim=1).item()
    
    labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    
    return {
        "prediction": labels[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            "Away Win": float(probabilities[0]),
            "Draw": float(probabilities[1]),
            "Home Win": float(probabilities[2])
        }
    }

# Function to run FastAPI server in a separate thread
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

# Start FastAPI server
@st.cache_resource
def start_fastapi():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Give server time to start
    return True

# Prediction function for Streamlit
def make_prediction(features):
    if model is None:
        return None, "Model not loaded"
    
    try:
        inputs = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(inputs)
        
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(outputs, dim=1).item()
        
        labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        
        return {
            "prediction": labels[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "probabilities": {
                "Away Win": float(probabilities[0]),
                "Draw": float(probabilities[1]),
                "Home Win": float(probabilities[2])
            }
        }, None
    except Exception as e:
        return None, str(e)

# Streamlit Frontend
def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">
        Predict football match outcomes using advanced machine learning
    </div>
    """, unsafe_allow_html=True)
    
    # Start FastAPI server
    start_fastapi()
    
    # Sidebar for information
    with st.sidebar:
        st.header(" About")
        st.info(
            "This app uses a neural network trained on football statistics to predict match outcomes.Proudly designed by Osinowo Abdulazeez "
            "Enter the team statistics below to get a prediction."
        )
        
        st.header(" Model Info")
        st.write(f"**Input Features:** {input_size}")
        st.write(f"**Hidden Layer Size:** {hidden_size}")
        st.write(f"**Output Classes:** 3 (Home Win, Draw, Away Win)")
        st.write(f"**Dropout Rate:** {dropout_prob}")
    
    # Feature input form
    st.header(" Match Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Home Team Stats")
        home_team = st.text_input("Home Team Name", placeholder="e.g., Manchester United")
        home_goals_scored = st.number_input("Total Goals Scored", min_value=0.0, step=0.1, key="home_scored")
        home_goals_conceded = st.number_input("Total Goals Conceded", min_value=0.0, step=0.1, key="home_conceded")
        #home_goal_ratio = st.number_input("Goal Ratio", min_value=0.0, step=0.01, key="home_ratio")
        home_points = st.number_input("Total Points", min_value=0.0, step=1.0, key="home_points")
        home_wins = st.number_input("Wins", min_value=0.0, step=1.0, key="home_wins")
        home_draws = st.number_input("Draws", min_value=0.0, step=1.0, key="home_draws")
        home_losses = st.number_input("Losses", min_value=0.0, step=1.0, key="home_losses")
    
    with col2:
        st.subheader(" Away Team Stats")
        away_team = st.text_input("Away Team Name", placeholder="e.g., Liverpool")
        away_goals_scored = st.number_input("Total Goals Scored", min_value=0.0, step=0.1, key="away_scored")
        away_goals_conceded = st.number_input("Total Goals Conceded", min_value=0.0, step=0.1, key="away_conceded")
        #away_goal_ratio = st.number_input("Goal Ratio", min_value=0.0, step=0.01, key="away_ratio")
        away_points = st.number_input("Total Points", min_value=0.0, step=1.0, key="away_points")
        away_wins = st.number_input("Wins", min_value=0.0, step=1.0, key="away_wins")
        away_draws = st.number_input("Draws", min_value=0.0, step=1.0, key="away_draws")
        away_losses = st.number_input("Losses", min_value=0.0, step=1.0, key="away_losses")
    
    # Prepare features
    features = [
        home_goals_scored, away_goals_scored, home_goals_conceded, away_goals_conceded,
        home_points, away_points,
        home_wins, home_draws, home_losses, away_wins, away_draws, away_losses,
        0.0, 0.0  # Placeholder for any additional features
    ]
    
    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(" Predict Match Outcome", use_container_width=True):
            if not home_team or not away_team:
                st.error("Please enter both team names!")
            else:
                with st.spinner("Making prediction..."):
                    result, error = make_prediction(features)
                
                if error:
                    st.error(f"Error making prediction: {error}")
                else:
                    # Display prediction results
                    st.success("Prediction completed!")
                    
                    # Match info
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 1.5rem; margin: 2rem 0;">
                        <strong>{home_team}</strong> 🆚 <strong>{away_team}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Main prediction
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    # Prediction card with gradient background
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="margin: 0;">🏆 Predicted Outcome</h2>
                        <h1 style="margin: 1rem 0; font-size: 3rem;">{prediction}</h1>
                        <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed probabilities
                    st.subheader(" Detailed Probabilities")
                    
                    prob_data = result["probabilities"]
                    
                    # Create three columns for probabilities
                    prob_cols = st.columns(3)
                    
                    outcomes = ["Home Win", "Draw", "Away Win"]
                    colors = ["#28a745", "#ffc107", "#dc3545"]
                    icons = ["🏠", "🤝", "✈️"]
                    
                    for i, (outcome, prob) in enumerate(prob_data.items()):
                        with prob_cols[i]:
                            st.markdown(f"""
                            <div style="
                                background-color: {colors[i]}20;
                                border-left: 4px solid {colors[i]};
                                padding: 1rem;
                                border-radius: 8px;
                                text-align: center;
                                margin: 0.5rem 0;
                            ">
                                <h3 style="margin: 0; color: {colors[i]};">{icons[i]} {outcome}</h3>
                                <h2 style="margin: 0.5rem 0; color: {colors[i]};">{prob:.2%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Feature summary
                    with st.expander(" Feature Summary"):
                        feature_names = [
                            "Home Goals Scored", "Away Goals Scored", "Home Goals Conceded", 
                            "Away Goals Conceded", "Home Goal Ratio", "Away Goal Ratio",
                            "Home Points", "Away Points", "Home Wins", "Home Draws", 
                            "Home Losses", "Away Wins", "Away Draws", "Away Losses",
                            "Extra Feature 1", "Extra Feature 2"
                        ]
                        
                        df = pd.DataFrame({
                            "Feature": feature_names,
                            "Value": features
                        })
                        st.dataframe(df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        Made using Streamlit and PyTorch | 
        <a href="http://127.0.0.1:8000" target="_blank">API Documentation</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
