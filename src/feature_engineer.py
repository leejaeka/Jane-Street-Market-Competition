def clean_data(df):
	# REQUIRMENT: nan_neg.csv and nan_pos.csv
	df = df.set_index('ts_id', drop=True)
	df.drop(columns=[f"resp_{i}" for i in range(1, 5)], inplace=True)
	print(f'Done loading data. df shape is {df.shape}')
	TARGET = 'resp'
	FEATURES = [f"feature_{i}" for i in range(1, 130)]
	train_pos, train_neg = df.loc[df.feature_0 > 0], df.loc[df.feature_0 < 0]
	train_pos.drop(columns=[TARGET, 'feature_0'], inplace=True)
	train_neg.drop(columns=[TARGET, 'feature_0'], inplace=True)
	gc.collect()
	nan_neg = pd.read_csv("../input/nan_neg.csv", header=None, sep=' ').values.astype(int)
	nan_pos = pd.read_csv("../input/nan_pos.csv", header=None, sep=' ').values.astype(int)

	# Split into X and y
	from copy import deepcopy as dc
	X_pos = dc(train_pos[FEATURES].values)
	X_neg = dc(train_neg[FEATURES].values)
	del train_pos, train_neg
	gc.collect()

	# load files 
	file = 'knn_5'
	path = "../input/janestreetimputeddata/"
	X_pos[nan_pos[0], nan_pos[1]] = pd.read_csv(path+f"positive_{file}.csv", 
												header=None, sep=' ').values.flatten()
	X_neg[nan_neg[0], nan_neg[1]] = pd.read_csv(path+f"negative_{file}.csv",                                         
												header=None, sep=' ').values.flatten()

	df = np.concatenate((X_pos, X_neg), axis=0)
	del X_pos, X_neg, nan_neg, nan_pos
	gc.collect()

	df = pd.DataFrame(df, columns = FEATURES)

	#hide_input
	train = dt.fread('../input/jane-street-market-prediction/train.csv')
	train = train.to_pandas()
	train = train[['date', 'weight', 'ts_id', 'resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4','feature_0']]
	gc.collect()

	# split train into 1s and 0s 
	upper = train[train['feature_0'] == 1].sort_values(by='ts_id', axis=0, ascending=True)
	lower = train[train['feature_0'] == -1].sort_values(by='ts_id', axis=0, ascending=True)

	# attach
	train = pd.concat([upper, lower], axis = 0)
	del upper, lower
	gc.collect()

	# save files
	df.to_csv('imputed.csv', index=False)
	train.to_csv('one_on_top.csv', index=False)
	gc.collect()

	df = pd.concat([train, imputed_df], axis=1, ignore_index=False)
	del train, imputed_df
	gc.collect()

	df = df.query('date > 85').reset_index(drop = True) 
	df = df[df['weight'] != 0]

	#hide_output
	# Add action column (this is our target)
	df['action'] = ((df['resp'].values) > 0).astype(int)

	# feature names
	features = [c for c in df.columns if "feature" in c]
	# resp names
	resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

	# We don't need time, date and weight anymore
	df = df.loc[:, df.columns.str.contains('feature|resp', regex=True)]

	# Get log transformation for tag groups
	tag_0_group = [9, 10, 19, 20, 29, 30, 56, 73, 79, 85, 91, 97, 103, 109, 115, 122, 123]
	for col in tag_0_group:
		df[str('log_'+str(col))] = (df[str('feature_'+str(col))]-df[str('feature_'+str(col))].min()+1).transform(np.log)

	return df