# trains and saves the coffee rating models

import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/"
    "refs/heads/main/data/coffee_analysis.csv"
)

# ordered from lightest to darkest roast
ROAST_CATEGORIES = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


# roast label -> number (NaN if unknown)
def roast_category(roast):
    return ROAST_CATEGORIES.get(roast, np.nan)


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def main():
    df = pd.read_csv(DATA_URL)
    y = df["rating"]

    # Exercise 1: linear regression on price
    lr = LinearRegression()
    lr.fit(df[["100g_USD"]].values, y)
    save_model(lr, "model_1.pickle")

    # Exercise 2: decision tree on price and roast category
    df["roast_cat"] = df["roast"].map(roast_category)
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(df[["100g_USD", "roast_cat"]].values, y)
    save_model(dtr, "model_2.pickle")

    # Bonus Exercise 4: TF-IDF text features -> linear regression.
    # The vectorizer is saved inside the pipeline, so words not seen in
    # training are simply ignored at prediction time.
    text_model = make_pipeline(TfidfVectorizer(), LinearRegression())
    text_model.fit(df["desc_3"].fillna(""), y)
    save_model(text_model, "model_3.pickle")


if __name__ == "__main__":
    main()
