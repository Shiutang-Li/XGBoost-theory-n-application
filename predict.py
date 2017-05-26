# coding=utf8
# description     : Use XGBoost model built to predict house prices
# author          : Shiu-Tang Li
# last update     : 05/26/2017
# version         : 0.1
# usage           : python predict.py
# python_version  : 3.5.2

import pandas as pd
import numpy as np
import xgboost as xgb


def main():
    # load data
    test = pd.read_csv('preprocessed_testing_data.csv')
    test_id = test['id']
    del test['price_doc']
    del test['id']
    X_test = xgb.DMatrix(data=test)

    # load model with parameters
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
    model = xgb.Booster(params=params, model_file='xgboost_for_moscow_house_prices.model')

    # prediction and output predicted results
    y_test = np.exp(model.predict(X_test)) - 1
    result = pd.DataFrame({'id': test_id, 'price_doc': y_test})
    result.to_csv("result.csv", index=False)
    print("predicted result output as 'result.csv'")

if __name__ == "__main__":
    main()
