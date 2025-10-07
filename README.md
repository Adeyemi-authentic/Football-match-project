#  FOOTBALL-MATCH

A machine learning project that predicts football match outcomes across Europe's top leagues, combining data science with 14 years of football watching experience.

##  Project Overview

This project leverages machine learning to predict match outcomes (Home Win, Draw, Away Win) for major European football leagues. The system processes historical match data and team statistics to generate predictions with **71% accuracy**.

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
- **Storage**: CSV files → MySQL database
- **Coverage**: Major European leagues with comprehensive match statistics

## 🧹 Data Preprocessing

### Data Cleaning
- ✅ Data type validation and formatting
- ✅ Null value handling (imputation/removal based on context)
- ✅ Error detection and correction

### Feature Engineering
- Created new features through statistical calculations
- Applied encoding for categorical variables
- **Target Variable Encoding**:
  - `0`: Away Win
  - `1`: Draw  
  - `2`: Home Win

## Machine Learning Pipeline

### Feature Selection
- **Initial Features**: 28
- **Selected Features**: 16-14
- **Selection Method**: 
  - scikit-learn's `SelectKBest`
  - Domain knowledge from football expertise

### Model Performance

| Model | Initial Accuracy | After Hyperparameter Tuning |
|-------|-----------------|----------------------------|
| **Random Forest** | 68% | 71% |
| **XGBoost** | 71% | 71% |
| **Neural Network (PyTorch)** | 65% | 68% |

### PyTorch Model Evaluation

#### Class Distribution
- **Class 0 (Away Win)**: 208 instances
- **Class 1 (Draw)**: 187 instances
- **Class 2 (Home Win)**: 316 instances

#### Performance Metrics (After Class Weight Balancing)

| Class | Precision | Recall | Notes |
|-------|-----------|--------|-------|
| **Away Win (0)** | 65% | 70% | Good balance between precision and recall |
| **Draw (1)** | 56% | 55% | Improved from 32% to 55% recall after balancing |
| **Home Win (2)** | 77% | 71% | Best performing class |

**Key Improvements:**
- Class weight balancing significantly improved Draw detection (recall: 32% → 55%)
- Draw class shows 56% precision, meaning when the model predicts a draw, it's correct 56% of the time
- Home Win predictions are most reliable with 77% precision

### Model Insights
- **Primary Metric**: Accuracy
- **Additional Metrics**: Precision, Recall, F1-Score
- **Validation**: Performance expected to improve with additional historical data
- **Class Imbalance**: Successfully addressed through class weight balancing

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
- [ ] Implement ensemble methods combining multiple models
- [ ] Real-time data integration via API webhooks
- [ ] Mobile application development
- [ ] Add head-to-head historical records
- [ ] Include weather conditions impact
- [ ] Track home/away form separately

##  Installation & Setup

```bash
# Clone the repository
git clone [repository-url]
cd FOOTBALL-DEPLOY

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials

# Run the Streamlit application
streamlit run app.py

# Or run the FastAPI backend
uvicorn main:app --reload
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



## API Documentation

Visit `/docs` endpoint when running the FastAPI server for interactive API documentation (Swagger UI), or `/redoc` for alternative documentation format.

Example: `http://localhost:8000/docs`

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

For questions or collaboration opportunities, please open an issue or reach out via [adey004azeez@gmail.com].

---

**Note**: This model is for educational and entertainment purposes. Always gamble responsibly if using predictions for betting purposes.# Football-match-project