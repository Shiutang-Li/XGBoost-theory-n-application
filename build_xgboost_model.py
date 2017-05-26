# coding=utf8
# description     : Build an XGBoost prediction model from preprocessed_training_data.csv
# author          : Shiu-Tang Li
# last update     : 05/26/2017
# version         : 0.1
# usage           : python build_xgboost_model.py
# python_version  : 3.5.2

import pandas as pd
import numpy as np
import xgboost as xgb
import gc


def main():
    # load data
    train = pd.read_csv('preprocessed_training_data.csv')
    y_train = np.log(train['price_doc'] + 1)
    del train['id']
    del train['price_doc']
    X_y_train = xgb.DMatrix(data=train, label=y_train)

    # garbage collection
    gc.collect()

    # take a look at the shape of data
    print("Training data shape:" + str(train.shape))  # X_train

    # set up parameters
    params = {
        'base_score': 15.7,
        'eta': 0.1,
        'colsample_bytree': 1,
        'max_depth': 7,
        'subsample': 1,
        'seed': 0,
        'lambda': 0,
        'gamma': 0,
        'objective': 'reg:linear',
        'eval_metric': 'rmse'
    }
    model = xgb.train(params=params,
                      dtrain=X_y_train,
                      num_boost_round=500)
    model.save_model('xgboost_for_moscow_house_prices.model')
    print("model saved as 'xgboost_for_moscow_house_prices.model'")

if __name__ == "__main__":
    main()
