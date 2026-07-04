# FIFA World Cup Prediction

## Version 1 Project Plan

---

# Project Objective

The objective of Version 1 is to build an end-to-end machine learning pipeline that predicts FIFA World Cup match outcomes and estimates each team's probability of winning the tournament.

The project will:

- Collect football data from free public data sources.
- Build a historical training dataset.
- Clean, preprocess, and integrate multiple datasets.
- Engineer football performance features.
- Train and compare baseline machine learning models.
- Predict match outcomes.
- Simulate the FIFA World Cup using Monte Carlo Simulation.
- Build an interactive Streamlit dashboard.
- Document the complete development process on GitHub and Substack.

---

# Scope

Version 1 focuses on building a complete baseline prediction system using publicly available football data.

## Data Sources

- Historical International Match Results
- FIFA World Rankings
- World Football Elo Ratings
- FIFA World Cup Historical Records
- FIFA World Cup Fixtures

## Machine Learning Models

- Logistic Regression
- Random Forest
- XGBoost

The best-performing model will be selected based on evaluation metrics.

## Dashboard Features

The dashboard will display:

- Team Strength Rankings
- Match Predictions
- Tournament Winner Probabilities
- Monte Carlo Simulation Results
- Model Performance Metrics
- Feature Importance

## Deliverables

The final Version 1 deliverables include:

- Automated data collection pipeline
- Feature engineering pipeline
- Machine learning prediction model
- Tournament simulation engine
- Interactive Streamlit dashboard
- GitHub repository
- Project documentation

# System Architecture

The FIFA World Cup Prediction system will follow the pipeline below:

Data Sources
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Model
        │
        ▼
Match Prediction
        │
        ▼
Monte Carlo Tournament Simulation
        │
        ▼
Interactive Streamlit Dashboard

## Data Sources

Version 1 will use five primary datasets.

| Dataset | Purpose | Collection Method | Refresh Frequency |
|----------|---------|-------------------|-------------------|
| Historical International Match Results | Train the machine learning model and calculate team performance metrics | Initial download + future automated updates | Periodically |
| FIFA World Rankings | Measure official team strength | Automated Python download | Monthly |
| World Football Elo Ratings | Measure dynamic team strength | Automated Python download | Daily |
| FIFA World Cup Historical Records | Calculate tournament experience features | One-time download | Rarely |
| FIFA World Cup Fixtures | Simulate the current tournament | Automated Python download | During the tournament |
## Feature Mapping

The following table documents every feature used in the machine learning model, its source, how it is calculated, and whether it is included in Version 1.

| Feature | Category | Source Dataset | Raw Variables Required | Engineered | Version 1 | Notes |
|---------|----------|----------------|------------------------|------------|------------|------|
| FIFA Rank | Team Strength | FIFA World Rankings | Rank | No | ✅ | Lower rank indicates stronger team |
| FIFA Points | Team Strength | FIFA World Rankings | Points | No | ✅ | Official FIFA rating |
| Elo Rating | Team Strength | World Football Elo Ratings | Elo Rating | No | ✅ | Dynamic team strength |
| Win Percentage | Recent Form | Historical Match Results | Match Result | Yes | ✅ | Last 10 matches |
| Goals Scored per Match | Attack | Historical Match Results | Goals Scored | Yes | ✅ | Last 10 matches |
| Goals Conceded per Match | Defense | Historical Match Results | Goals Conceded | Yes | ✅ | Last 10 matches |
| Goal Difference | Attack/Defense | Historical Match Results | Goals Scored, Goals Conceded | Yes | ✅ | Last 10 matches |
| Clean Sheet Percentage | Defense | Historical Match Results | Goals Conceded | Yes | ✅ | Last 10 matches |
| Average Opponent Elo | Strength of Schedule | Historical Matches + Elo | Opponent, Elo | Yes | ✅ | Average opponent quality |
| Average Opponent FIFA Rank | Strength of Schedule | Historical Matches + FIFA | Opponent Rank | Yes | ✅ | Average opponent ranking |
| World Cup Appearances | Experience | World Cup History | Appearances | No | ✅ | Tournament experience |
| World Cup Championships | Experience | World Cup History | Titles | No | ✅ | Winning pedigree |
| Best World Cup Finish | Experience | World Cup History | Best Finish | Yes | ✅ | Converted to numeric score |
| Host Nation | Match Context | World Cup Fixtures | Host Country | Yes | ✅ | Binary feature |
| Neutral Venue | Match Context | Historical Match Results | Neutral Venue | Yes | ✅ | Binary feature |
| Confederation | Geography | FIFA World Rankings | Confederation | No | ✅ | UEFA, CONMEBOL, etc. |
| Recent Form Score | Engineered | Calculated | Win %, Goal Difference | Yes | ✅ | Composite score |
| Attack Score | Engineered | Calculated | Goals/Game, Goal Difference | Yes | ✅ | Composite score |
| Defense Score | Engineered | Calculated | Goals Conceded, Clean Sheets | Yes | ✅ | Composite score |
| Experience Score | Engineered | Calculated | Appearances, Titles, Best Finish | Yes | ✅ | Composite score |
| Team Strength Score | Dashboard | Calculated | Elo, FIFA, Form, Experience | Yes | Dashboard Only | Used for visualization, not training |

## Feature Engineering

Feature engineering transforms raw football data into meaningful variables that improve the predictive performance of the machine learning models.

The following engineered features will be created during Version 1.

| Feature | Formula / Method | Reason |
|----------|------------------|--------|
| Win Percentage | Wins ÷ Last 10 Matches | Measures recent performance |
| Goals Scored per Match | Total Goals Scored ÷ Matches Played | Measures attacking ability |
| Goals Conceded per Match | Total Goals Conceded ÷ Matches Played | Measures defensive ability |
| Goal Difference | Goals Scored − Goals Conceded | Overall team quality |
| Clean Sheet Percentage | Clean Sheets ÷ Matches Played | Defensive consistency |
| Average Opponent Elo | Mean Elo Rating of Opponents | Measures strength of schedule |
| Average Opponent FIFA Rank | Mean FIFA Rank of Opponents | Additional schedule strength |
| Recent Form Score | Weighted combination of Win %, Goal Difference and Goals/Game | Measures current momentum |
| Attack Score | Weighted attacking metrics | Overall offensive capability |
| Defense Score | Weighted defensive metrics | Overall defensive capability |
| Experience Score | Weighted combination of World Cup appearances, titles and best finish | Tournament experience |
| Team Strength Score | Weighted combination of Elo, FIFA Rank, Form and Experience | Dashboard visualization only |

## Machine Learning Pipeline

Version 1 will compare multiple supervised machine learning models to determine which model best predicts international football match outcomes.

### Workflow

Historical Data

↓

Data Cleaning

↓

Feature Engineering

↓

Train/Test Split

↓

Model Training

↓

Model Evaluation

↓

Best Model Selection

↓

Match Prediction

↓

Tournament Simulation

### Models

- Logistic Regression
- Random Forest
- XGBoost

The same training dataset and engineered features will be used for each model to ensure a fair comparison.

## Model Evaluation

The following evaluation metrics will be used to compare machine learning models.

| Metric | Purpose |
|---------|---------|
| Accuracy | Overall prediction accuracy |
| Precision | Correct positive predictions |
| Recall | Ability to identify true positives |
| F1 Score | Balance between Precision and Recall |
| ROC-AUC | Measures model discrimination |
| Log Loss | Evaluates probability predictions |

The best-performing model will be selected based on overall performance across these evaluation metrics rather than a single metric.

## Tournament Simulation

After predicting match probabilities, the selected machine learning model will be used to simulate the FIFA World Cup tournament.

Simulation Process

1. Predict the probability of every match.
2. Simulate each match outcome using predicted probabilities.
3. Advance the winning team through the tournament bracket.
4. Repeat the tournament thousands of times using Monte Carlo Simulation.
5. Calculate the probability of each team reaching every tournament stage.

### Simulation Output

- Group Stage Qualification Probability
- Round of 16 Probability
- Quarter-final Probability
- Semi-final Probability
- Final Probability
- Championship Probability

## Dashboard Design

The Streamlit dashboard will present predictions and simulation results through an interactive web application.

### Dashboard Pages

### 1. Tournament Overview

Displays:

- Team Strength Rankings
- Championship Probabilities
- Top Tournament Contenders

### 2. Match Predictor

Users can:

- Select Team A
- Select Team B

Outputs:

- Win Probability
- Draw Probability
- Loss Probability

### 3. Tournament Simulation

Displays:

- Tournament Bracket
- Stage Advancement Probabilities
- Championship Probability

### 4. Model Performance

Displays:

- Model Comparison
- Feature Importance
- Confusion Matrix
- Evaluation Metrics

## Repository Roadmap

| Stage | Status |
|---------|---------|
| Project Setup | ✅ Complete |
| Data Collection | ⬜ Planned |
| Data Cleaning | ⬜ Planned |
| Feature Engineering | ⬜ Planned |
| Model Training | ⬜ Planned |
| Model Evaluation | ⬜ Planned |
| Tournament Simulation | ⬜ Planned |
| Dashboard Development | ⬜ Planned |
| Deployment | ⬜ Planned |

## Future Improvements

Version 2

- Additional engineered features
- Rolling performance metrics
- Hyperparameter tuning
- Automated feature selection

Version 3

- Squad market values
- Player ratings
- Expected Goals (xG)
- Injury information

Version 4

- Automated data refresh pipeline
- CI/CD deployment
- Docker containerization
- Cloud deployment
- Real-time prediction updates
