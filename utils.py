import numpy as np


def check_nan(df):
    columns = df.columns
    nan_list = list()
    for s in columns:
        num = sum(df[s].isnull()) / df.shape[0] * 100
        if num:
            nan_list.append((s, num))
    return nan_list


def normalize(df, columns):
    for s in columns:
        df[s] = (df[s] - df[s].mean()) / (df[s].max() - df[s].min())
    return df


def fill_empty(df, columns):
    for s in columns:
        if s[0].endswith('cat') or s[0].endswith('bin'):
            df[s[0]] = df[s[0]].fillna(df[s[0]].mode()[0])
        else:
            df[s[0]] = df[s[0]].fillna(df[s[0]].mean())
    return df


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred):
    assert(len(actual) == len(pred))
    all_ = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all_ = all_[np.lexsort((all_[:, 2], -1*all_[:, 1]))]
    total_losses = all_[:, 0].sum()
    gini_sum = all_[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# Create an XGBoost-compatible metric from Gini
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]
