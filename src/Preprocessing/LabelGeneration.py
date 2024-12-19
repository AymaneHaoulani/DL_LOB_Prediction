import pandas as pd
import numpy as np

# --- Label Encoding ---

# This function returns the encoded midprice variation column.
def get_midprice_variation_column(data, horizon, theta):
        # Compute the midprice
    data['Midprice'] = (data['Bid Price 1'] + data['Ask Price 1']) / 2
        # Get the mean modprice over a specified horizon/window
    future_means = data['Midprice'].shift(-1).rolling(window=horizon, min_periods=1).mean()
        # Relative distance to the future means
    variation = (future_means - data['Midprice']) / data['Midprice'] * 100
        # Categorize rows depending on the comparison between the computed variation and theta
    data['label'] = pd.cut(variation, bins=[-np.inf, -theta, theta, np.inf], labels=['D', 'S', 'U'])
    return data['label']

# This function encode the label column D : 0, S : 1, U : 2
def label_encoding(label_column):
    label_mapping = {'D': 0, 'S': 1, 'U': 2}
    label_column = label_column.map(label_mapping)
    return label_column

# Method that returns the optimal theta
def get_best_theta(k):
    if k == 100:
        return 6.154545*1e-4
    else:
        return 0.1*1e-4
    