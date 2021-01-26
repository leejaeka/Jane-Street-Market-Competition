import janestreet
env = janestreet.make_env()
th = 0.5

f = np.median
models = models[-3:]

def feature_engineering(df):
    tag_0_group = [9, 10, 19, 20, 29, 30, 56, 73, 79, 85, 91, 97, 103, 109, 115, 122, 123]
    for col in tag_0_group:
        df['log_'+str(col)] = (df['feature_'+str(col)]-df['feature_'+str(col)].min()+1).transform(np.log)
    return df

if __name__ == "__main__":
	for (test_df, pred_df) in tqdm(env.iter_test()):
		if test_df['weight'].item() > 0:
			x_tt = test_df.loc[:, features]
			x_tt = feature_engineering(x_tt).values
			if np.isnan(x_tt[:, 1:].sum()):
				x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
			pred = np.mean([model(x_tt, training = False).numpy() for model in models],axis=0)
			pred = f(pred)
			pred_df.action = np.where(pred >= th, 1, 0).astype(int)
		else:
			pred_df.action = 0
		env.predict(pred_df)