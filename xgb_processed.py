import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import utils


# Reading the test and train data
df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')
print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)

# Storing columns that are not required for training
y_train = df_train['target'].values
id_train = df_train['id'].values
id_test = df_test['id'].values

# Dropping columns that are not required for training
x_train = df_train.drop(['target', 'id'], axis=1)
x_test = df_test.drop(['id'], axis=1)

# Replacing -1 with NaN
x_train = x_train.replace(-1, np.NaN)
x_test = x_test.replace(-1, np.NaN)

# Checking each column for NaN
train_nan = utils.check_nan(x_train)
test_nan = utils.check_nan(x_test)

# Fill nan
x_train = utils.fill_empty(x_train, train_nan)
x_test = utils.fill_empty(x_test, test_nan)
print(utils.check_nan(x_train), utils.check_nan(x_test))

# Normalizing data
cont = [s for s in x_train.columns if (not s.endswith('cat')) or (not s.endswith('bin'))]
x_train = utils.normalize(x_train, cont)
x_test = utils.normalize(x_test, cont)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
print('Train samples: {}, Validation samples; {}'.format(len(x_train), len(x_test)))

# Up sampling

# converting to xgboost matrix format
d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(x_test)

params = dict()
params['objective'] = 'binary:logistic'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 6
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
mdl = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,
                feval=utils.gini_xgb, maximize=True, verbose_eval=10)

# Predict on our test data
p_test = mdl.predict(d_test)

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = p_test
sub.to_csv('input/submissions/xgb1.csv', index=False)

print(sub.head())
