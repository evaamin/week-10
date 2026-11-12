import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train import ROAST_CATEGORIES, roast_category

# pickles live next to this file
MODEL_DIR = Path(__file__).parent


# load a pickled model by filename
def _load(name):
    with open(MODEL_DIR / name, "rb") as f:
        return pickle.load(f)


# predict ratings from price + roast, or from review text if text=True
def predict_rating(df_X, text=False):
    # text mode: use the TF-IDF model
    if text:
        # accept a one-column dataframe or a list/series of strings
        if isinstance(df_X, pd.DataFrame):
            texts = df_X.iloc[:, 0]
        else:
            texts = pd.Series(df_X)
        return _load("model_3.pickle").predict(texts.fillna("").astype(str))

    price = df_X["100g_USD"].to_numpy(dtype=float)
    # which rows have a roast seen in training
    known = df_X["roast"].isin(ROAST_CATEGORIES.keys()).to_numpy()
    y_pred = np.empty(len(df_X), dtype=float)

    # known roast: decision tree on price + roast
    if known.any():
        roast_cat = df_X["roast"][known].map(roast_category).to_numpy()
        X2 = np.column_stack([price[known], roast_cat])
        y_pred[known] = _load("model_2.pickle").predict(X2)

    # unknown roast: linear regression on price only
    if (~known).any():
        X1 = price[~known].reshape(-1, 1)
        y_pred[~known] = _load("model_1.pickle").predict(X1)

    return y_pred
