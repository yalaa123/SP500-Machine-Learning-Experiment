--Features--
- Calculates daily S&P 500 returns
- Uses previous returns to predict the next day's return
- Measures directional prediction accuracy
- Predicts next-day market volatility
- Predicts average volatility over the next five days
- Evaluates models using R² score and correlation
- Compares predictions against a simple baseline model

--Files--
- `ML.py` — main analysis and machine learning script
- `SP500.csv` — historical S&P 500 data

--Requirements--

- pandas
- NumPy
- scikit-learn

--Experiments--
- Uses the previous three days' returns to predict the next day's return.
- Uses previous absolute returns and rolling averages to predict next-day volatility.
- Predicts the average absolute return over the next five days.

--Results and Findings--
- The experiments showed limited predictive power.
- The return prediction model struggled to identify future market movements.
- The next-day volatility model showed very weak correlation between predicted and actual volatility.
- The five-day volatility model performed slightly better than the simple 20-day average baseline, but still produced a negative R² score.

--Conclusion--
The experiment did not produce a useful predictive model, but provided experience with data analysis, feature engineering, Linear Regression, and model evaluation.
