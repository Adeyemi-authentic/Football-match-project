#  FOOTBALL-MATCH

A machine learning project that predicts football match outcomes across Europe's top leagues, combining data science with 14 years of football watching experience.

**Try it live:**
- **Web app:** https://football-predictor-m14h.onrender.com
- **API:** https://football-predictor-api-usho.onrender.com (interactive docs at [`/docs`](https://football-predictor-api-usho.onrender.com/docs))

Both run on Render's free plan, so the first visit after a quiet spell can take up to a minute while the service wakes up.

##  Project Overview

This project leverages machine learning to predict match outcomes (Home Win, Draw, Away Win) for major European football leagues. The system processes historical match data and team statistics to generate predictions.

## Leagues Covered

- **Premier League** (England)
- **La Liga** (Spain) 
- **Serie A** (Italy)
- **Bundesliga** (Germany)
- **Eredivisie** (Netherlands)
- **Ligue 1** (France)
- **Primeira Liga** (Portugal)

## Project Architecture

```
Football Prediction System
├──  Data Collection (ETL)
├──  Data Preprocessing
├──  Machine Learning Pipeline
├──  Model Deployment
└──  Automation
```

## Data Collection

- **Data Source**: FootballAPI.com
- **Time Period**: 2022/2023 season to present
- **Storage**: `data/matches_data.csv` → MySQL database (`matches_data` table)
- **Coverage**: Major European leagues with comprehensive match statistics

## Data Preprocessing

### Data Cleaning
- ✅ Data type validation and formatting
- ✅ Null value handling (imputation/removal based on context)
- ✅ Error detection and correction

### Feature Engineering
Matches are replayed in date order and every feature is taken from each team's state **before kick-off**, so features never include the result being predicted. 14 features (7 per team):

| Feature | Description |
|---|---|
| `elo` | Elo rating (K=15, 60-point home advantage, goal-margin multiplier, carried over between seasons; promoted teams start 50 points below 1500) |
| `ppg` | Points per game this season |
| `gf_pg` / `ga_pg` | Goals scored / conceded per game this season |
| `venue_ppg` | Points per game at home (home team) or away (away team) this season |
| `form_ppg` / `form_gd` | Points and goal difference per game over the last 5 matches |

Per-game rates are smoothed towards typical league values for teams with few games played (see `features.py`).
- **Target Variable Encoding**:
  - `0`: Away Win
  - `1`: Draw  
  - `2`: Home Win

## Machine Learning Pipeline

### Model Performance

Measured on the 1,043 most recent matches (16 March to 19 October 2026), which no model saw during training or tuning. Data: 5,215 matches from the 2024/25, 2025/26 and 2026/27 seasons.

| Model | Accuracy | Log loss (lower is better) |
|-------|----------|----------------------------|
| Always predict home win | 42.8% | – |
| Training class frequencies | 42.8% | 1.076 |
| **Logistic regression (used by the app)** | **50.7%** | **0.999** |
| Gradient boosting | 50.7% | 1.009 |
| Neural network (PyTorch) | 48.0% | 1.015 |

The app uses logistic regression because it had the best log loss and accuracy. Gradient boosting matched its accuracy with worse log loss, and the neural network did worse, so the more complex models did not find anything the linear model misses on these 14 features. The Elo difference alone carries almost all of the signal.

### Model Insights
- About 8 points better than always picking the home team. Bookmaker odds typically reach 50–53%, so this is a realistic result.
- The model almost never names a draw as the most likely result (draws are rarely more likely than both wins), but the draw probability it shows is still meaningful.
- Earlier versions of this README reported 71%. That figure came from a data-leakage bug (features included the result of the match being predicted) and has been withdrawn.

## Deployment

### Technology Stack
- **Frontend**: Streamlit
- **Backend API**: FastAPI
- **Hosting**: Render
- **Database**: MySQL
- **Experiment Tracking**: Weights & Biases (wandb)

### Features
-  Interactive web application
- Real-time predictions
-  RESTful API endpoints
- Responsive design
-  Model performance tracking

## Future Improvements

- [ ] Expand dataset with more historical seasons (3-5 years)
- [ ] Include player-level statistics
- [ ] Add injury and suspension data
- [ ] Real-time data integration via API webhooks
- [ ] Mobile application development
- [ ] Add head-to-head historical records
- [ ] Include weather conditions impact
- [ ] Track home/away form separately

##  Installation & Setup

```bash
# Clone the repository
git clone [repository-url]


# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials

# Scrape data (writes data/matches_data.csv and uploads to MySQL)
python data_scraped_storage.py

# Train the model (writes model.json: coefficients, scaler and team data)
python train.py

# Run the Streamlit application
streamlit run football_predictor_app.py

# Or run the FastAPI backend
uvicorn app:app --reload
```

## Requirements

```txt
pandas>=1.5.0
scikit-learn>=1.2.0
xgboost>=1.7.0
torch>=2.0.0
streamlit>=1.20.0
fastapi>=0.95.0
uvicorn>=0.20.0
mysql-connector-python>=8.0.0
requests>=2.28.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
wandb>=0.15.0
python-dotenv>=1.0.0
```

## Usage

### Making Predictions via Web Interface
1. Navigate to the deployed Streamlit app
2. Select the league and teams
3. View prediction probabilities for each outcome



## Evaluation

`train.py` holds out the most recent 20% of matches as a test set (time-based, not random), uses early stopping on a validation slice before that, and picks each model's settings on a validation slice before that. It prints accuracy and log loss for logistic regression, gradient boosting and a neural network next to simple baselines (always home win, class frequencies). The saved model is the logistic regression, refitted on all matches. Realistic accuracy for match outcome prediction is around 50–55%.

## Deploying on Render

`render.yaml` defines the Streamlit app and the API as two free web services with auto-deploy off. In Render, create a **Blueprint** from the repo, then use **Manual Deploy → Deploy latest commit** after checking a retrained model. The services install `requirements-render.txt`, which has no PyTorch or scikit-learn: the app computes predictions from the saved coefficients with numpy. `model.json` must be committed, since it contains the coefficients and the team data the app predicts from.

## API Documentation

- `GET /teams` lists teams per competition (latest season)
- `POST /predict` takes `{"home_team": "...", "away_team": "..."}` and returns the prediction, probabilities and the features used

Visit `/docs` endpoint when running the FastAPI server for interactive API documentation (Swagger UI), or `/redoc` for alternative documentation format.

Example: `http://localhost:8000/docs`, or the live docs at https://football-predictor-api-usho.onrender.com/docs

```bash
curl -X POST https://football-predictor-api-usho.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Chelsea FC", "away_team": "AFC Bournemouth"}'
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License
## Author

**Abdulazeez Oluwafimihan**
- A Data scientist , aspiring ML engineer and a Passionate Chelsea fan 


##  Acknowledgments

- FootballAPI.com for providing comprehensive match data
- The open-source community for excellent ML libraries
- All contributors and football enthusiasts who provided feedback
- Special thanks to Claude/chatgpt for assistance,learning and guidance.

## Contact

For questions or collaboration opportunities, please open an issue or reach out via  email adey004azeez@gmail.com.

---

**Note**: This model is for educational and entertainment purposes. Always gamble responsibly if using predictions for betting purposes.# Football-match-project
