import pandas as pd


def preprocess(dataset):
    if "zipcode" in dataset.columns:
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