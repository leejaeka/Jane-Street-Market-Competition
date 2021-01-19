# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import gc
from time import time
import multiprocessing

import numpy as np
import pandas as pd

import xgboost as xgb
import janestreet

# Load data
data = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
data=data[data.weight!=0]

# Settings
NAN_VALUE = -9999
features = [c for c in data.columns if 'feature' in c]
target = 'resp'

# Split into features X and target Y
X = data.loc[:, features].fillna(NAN_VALUE)
Y = (data.loc[:, target] > 0).astype(int)

# Clear memory
del data
gc.collect()

# Train model
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=11,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    missing=NAN_VALUE,
    random_state=2020,
    tree_method='gpu_hist',
    nthread=multiprocessing.cpu_count()
)
model.fit(X, Y)
print('Finished training model')

# Clear memory
del X, Y
gc.collect()

# Create submission
env = janestreet.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test: 
    X_test = test_df.loc[:, features]
    X_test = X_test.fillna(NAN_VALUE)
    sample_prediction_df.action = model.predict(X_test)
    env.predict(sample_prediction_df)

# SCORE: 6005.582