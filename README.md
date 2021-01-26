# Jane Street Market Prediction

[Part1 - EDA](https://leejaeka.github.io/jaekangai/python/eda/jane%20street/kaggle/visualization/big%20data/2021/01/23/JaneStreet-Copy1.html) <br>
[Part2 - Predict]()
[Part3 - Blog post]()

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

### INSTRUCTION although I don't recommend you to run this since this is specifically for Jane Street ANONYMIZED FEATURES competition..
input folder - Put the feature.csv, train.csv from https://www.kaggle.com/c/jane-street-market-prediction and imputed_df.csv from (https://www.kaggle.com/louise2001/janestreetimputeddata). <br>
src folder - scripts are stored here. Just run python train.py to clean and train and save model to models folder. submit.py is to submit for Kaggle competition. <br>
model folder - the models are saved here
output folder - the output cleaned data are saved here
notebook folder - notebooks used for EDA and feature engineering and training are here.
