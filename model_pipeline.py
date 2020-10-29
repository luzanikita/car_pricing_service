import os
import sys
from datetime import datetime

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


DATASET_PATH = "../kaggle/data/train.csv"
SAVE_DIR = "predictions"
RANDOM_SEED = 42


def main(dataset_path, save_dir):
    np.random.seed(RANDOM_SEED)
    X_train, X_test, y_train, y_test = prepare_data(dataset_path)
    model = train(X_train, y_train)
    save_model(model, save_dir)
    score = evaluate(model, X_test, y_test, mean_absolute_percentage_error)
    return score
    

def prepare_data(dataset_path):
    dataset = pd.read_csv(dataset_path, index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.drop(columns="price"),
        dataset.price,
        test_size=0.1,
        random_state=RANDOM_SEED
    )
    cond = y_train <= np.quantile(y_train, 0.95)
    X_train = X_train[cond]
    y_train = y_train[cond]

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    y_train = np.log(y_train)

    return X_train, X_test, y_train, y_test


def preprocess(dataset):
    dataset.drop(columns="zipcode", inplace=True)

    dataset = fix_year(dataset)
    dataset = process_categories(dataset)
    
    return dataset


def fix_year(dataset):
    dataset.registration_year = dataset.registration_year.apply(lambda y : 2000 + y if y < 21 else y)
    dataset.registration_year = dataset.registration_year.apply(lambda y : 1900 + y if y < 100 else y)
    return dataset


def process_categories(dataset):
    dataset['damage'] = dataset['damage'].fillna('empty')
    dataset['gearbox'] = dataset['gearbox'].fillna('manual')
    dataset['fuel'] = dataset['fuel'].fillna('empty')
    dataset['type'] = dataset['type'].fillna('empty')
    type_dict = {
        'bus':'bus', 'convertible':'coupé','coupé':'coupé', 
        'limousine':'limousine', 'other':'small car', 'empty':'wagon',
        'small car':'small car', 'station wagon':'wagon'
    }
    dataset['type'] = dataset['type'].map(type_dict)

    cat_features = ["type", "gearbox", "model", "fuel", "brand", "damage"]
    for col in cat_features:
        dataset[col] = dataset[col].astype('category')
    
    return dataset


def train(X_train, y_train):
    model = lgb.LGBMRegressor(
        random_state=RANDOM_SEED,
        objective='mape',
        num_leaves=100,
        feature_fraction=0.9,
        max_depth=-1,
        learning_rate=0.03,
        num_iterations=2000,
        subsample=0.5,
        categorical_feature="auto"
    )
    model.fit(X_train, y_train)

    return model


def predict(model, X):
    prediction = model.predict(X)
    return np.exp(prediction)


def evaluate(model, X, y_true, metric):
    y_pred = predict(model, X)
    return metric(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def save_model(model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(
        save_dir, 
        f"model_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.pkl"
    )
    with open(save_path, 'wb') as dst:
        pickle.dump(model, dst)


def load_model(model_path):
    with open(model_path, 'rb') as src:
        model = pickle.load(src)

    return model


def test_prediction():
    model = load_model("predictions/model_2020-10-22_22:03.pkl")
    data = pd.read_csv("../kaggle/data/test_no_target.csv", index_col=0)
    data = preprocess(data)
    predictions = predict(model, data)
    save_submisson(data, predictions)


def save_submisson(data, predictions):
    submission = pd.DataFrame({"Id": data.index, "Predicted": predictions})
    submission.sort_values("Id", inplace=True)
    filename = f"sumbission_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.csv"
    submission.to_csv(filename, index=False)


if __name__ == "__main__":
    score = main(DATASET_PATH, SAVE_DIR)
    print(score)
    test_prediction()
