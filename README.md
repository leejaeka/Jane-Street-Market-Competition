# Jane Street Market Prediction

Kaggle Competition. This project includes exploratory data analysis(notebooks) and prediction scripts.
### Problem: 130 anonymized features. Predict whether taking each trading opportunities will result in profit for a whole year.

### Dataset: 12GB of Real world financial markets. 
- anonymized set of features, feature_{0...129}, representing real stock market data.
- each row in the dataset represents a trading opportunity, for which you will be predicting an action value: 1 to make the trade and 0 to pass on it.
- each trade has an associated weight and resp, which together represents a return on the trade.
- date column is an integer which represents the day of the trade, while ts_id represents a time ordering.
- in addition to anonymized feature values, you are provided with metadata about the features in features.csv.
cred: https://www.kaggle.com/c/jane-street-market-prediction/data

### Modules used: numpy, pandas, tensorflow, tqdm, random, datatable, sklearn, gc, seaborn, matplotlib, plotly, defaultdict
