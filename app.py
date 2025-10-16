from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List


class FootballML(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply first linear layer and activation
        x = self.dropout(x)  # Apply dropout
        output = self.fc2(x)  # Final linear layer
        return output


input_size = 14
hidden_size = 36
dropout_prob = 0.35
output_dim = 3

model = FootballML(input_size, hidden_size, output_dim, dropout_prob)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Create FastAPI app
app = FastAPI()


class MatchFeatures(BaseModel):
    features: List[float]  # Fixed: Use List[float] instead of list[14.0]


@app.get("/")
def home():
    return {"message": "Football Predictor API is running"}


@app.post("/predict")
def predict(data: MatchFeatures):
    # Validate input size
    if len(data.features) != input_size:
        return {"error": f"Expected {input_size} features, got {len(data.features)}"}

    inputs = torch.tensor([data.features], dtype=torch.float32)

    with torch.no_grad():  # Added for inference efficiency
        outputs = model(inputs)

    predicted_class = torch.argmax(outputs, dim=1).item()
    labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    return {"prediction": labels[predicted_class]}  # Fixed: "prediction" not "predictions"