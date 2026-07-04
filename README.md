# ⚽ FIFA World Cup Prediction

An end-to-end data science project that predicts FIFA World Cup match outcomes and tournament winners using machine learning, historical international football data, FIFA rankings, World Football Elo ratings, World Cup history, feature engineering, and Monte Carlo simulation.

The model predicts individual match probabilities first. Then, a Monte Carlo simulation uses those probabilities to estimate each team's chance of winning the tournament.

## Features Used

### Team Strength
- FIFA Ranking
- FIFA Ranking Points
- World Football Elo Rating

### Recent Performance
- Win Percentage
- Goals Scored per Match
- Goals Conceded per Match
- Goal Difference
- Clean Sheet Percentage

### Strength of Schedule
- Average Opponent Elo Rating
- Average Opponent FIFA Ranking

### Tournament Experience
- World Cup Appearances
- World Cup Championships
- Best World Cup Finish

### Match Context
- Confederation
- Host Nation
- Neutral Venue

### Engineered Features
- Recent Form Score
- Attack Score
- Defense Score
- Experience Score

## Planned Improvements

Future versions of this project will test additional features and compare how each version improves model performance.

Planned improvements include:

- Recent Goal Difference
- Performance Against Top-Ranked Teams
- Tournament Momentum
- Squad Quality
- Rest Days Between Matches
- Model comparison across Logistic Regression, Random Forest, and XGBoost
- Interactive Streamlit dashboard
- Monte Carlo simulation output by tournament stage

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- GitHub

## Project Status

🚧 In development