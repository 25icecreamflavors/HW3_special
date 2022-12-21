import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from conf.conf import logging, settings
from connector.connector import get_data
from util.util import load_model, save_model


def split(df: pd.DataFrame) -> list:
    '''
    Function for splitting the dataset on train/test.
    '''
    logging.info('Splitting data')
    y = df['target'].values
    X = df.drop(columns=['target']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=settings.PARAMS.test_size,
                                                       random_state=settings.PARAMS.random_state)
    logging.info('Splitted the data!')
    return X_train, X_test, y_train, y_test


def grid(X_train:list, y_train:list, model, params:dict) -> dict:
    '''
    Method running gridsearch to tune hyperparams.
    '''
    logging.info('Starting gridsearch')
    clf = GridSearchCV(model, params)
    clf.fit(X_train, y_train)
    logging.info('Finished gridsearch successfully!')
    return clf.best_params_


def train_rf(X_train:list, y_train:list, params:dict = settings.PARAMS.rf_train) -> RandomForestClassifier:
    '''
    Methos for training the random forest. Runs gridsearch to choose hyperparams.
    '''
    logging.info('Started - RF model training')
    best_params = grid(X_train, y_train, RandomForestClassifier(), params)
    clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)
    logging.info('Finished - RF model training')
    save_model(clf, rf=True)
    return clf


def train_gb(X_train:list, y_train:list, params:dict = settings.PARAMS.gb_train) -> GradientBoostingClassifier:
    '''
    Methos for training the gradient boosting. Runs gridsearch to choose hyperparams.
    '''
    logging.info('Started - GB model training')
    best_params = grid(X_train, y_train, GradientBoostingClassifier(), params)
    clf = GradientBoostingClassifier(**best_params)
    clf.fit(X_train, y_train)
    logging.info('Finished - GF model training')
    save_model(clf, rf=False)
    return clf


def predict(values:list, model:str) -> list:
    '''
    Getting predictions from the choosen model.
    '''
    if model == 'rf':
        model_path = settings.MODEL.rf_path
        fl = 1
    else:
        model_path = settings.MODEL.gb_path
        fl = 0

    if os.path.exists(model_path):
        clf = load_model(model_path)
    else:
        df = get_data(settings.DATA.data_set)

        X_train, _, y_train, _ = split(df)
        if fl:
            clf = train_rf(X_train, y_train)
        else:
            clf = train_gb(X_train, y_train)

    return clf.predict(values)