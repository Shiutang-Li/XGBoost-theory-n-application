# coding=utf8
# description     : Preprocess train.csv and test.csv from "Kaggle
#                   Sberbank Russian Housing Market competition".
# author          : Shiu-Tang Li
# last update     : 05/26/2017
# version         : 0.1
# usage           : python preprocessing.py
# python_version  : 3.5.2

import pandas as pd
import numpy as np


def main():
    # load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print('Data loaded.')

    test['price_doc'] = np.nan
    train['label'] = 1
    test['label'] = 0
    data = pd.concat([train, test])

    # time features
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["year"] = data["timestamp"].dt.year
    del data['timestamp']

    # adjust weird full_sq, set it equal to life_sq
    temp = data['id'].values
    temp2 = data['full_sq'].values
    temp3 = data['life_sq'].values

    for i, item in enumerate(temp):
        if ((item == 1189) or (item == 2012) or
           (item == 11335) or (item == 22415) or
           (item == 24299) or (item == 26267) or
           (item == 26366) or (item == 30938) or
           (item == 35857)):
            temp2[i] = temp3[i]

    # remove full_sq outliers (may be wrong values)
    data = data[data['full_sq'] != 0]
    data = data[data['full_sq'] != 1]

    # if life_sq > full_sq, set life_sq = full_sq
    temp = data['life_sq'].values
    temp2 = data['full_sq'].values
    for i, item in enumerate(temp):
        if item > temp2[i]:
            temp[i] = temp2[i]

    # change typos (0,1) in life_sq to missing values
    temp = data['life_sq'].values
    temp2 = data['full_sq'].values
    for i, item in enumerate(temp):
        if (item == 0) or (item == 1):
            temp[i] = np.nan

    # if kitch_sq > life_sq, or kitch_sq > full_sq, set kitch_sq = np.nan
    temp = data['kitch_sq'].values
    temp2 = data['life_sq'].values
    temp3 = data['full_sq'].values
    for i, item in enumerate(temp):
        if (item > temp2[i]) or (item > temp3[i]):
            temp[i] = np.nan

    # remove outliers (full_sq / life_sq too big, or full_sq too big)
    data = data[data['id'] != 1481]
    data = data[data['id'] != 1613]
    data = data[data['id'] != 2428]
    data = data[data['id'] != 2783]
    data = data[data['id'] != 3530]
    data = data[data['id'] != 5947]
    data = data[data['id'] != 7210]
    # based on
    # data[(data['full_sq'] > 120) &
    #     ((data['life_sq'] / data['full_sq'])<=0.2) &
    #     (data['label'] ==1)][['id', 'full_sq', 'life_sq','kitch_sq','price_doc']].head(20)

    # remove outliers - price_doc too high
    data = data[data['id'] != 2121]
    # remove outliers - price_per_sqm too high
    data = data[data['id'] != 6115]

    # build year preprocessing
    temp = data['build_year'].values
    for i, item in enumerate(temp):
        if ((item == 0) or (item == 1) or (item == 2) or
           (item == 3) or (item == 71) or (item == 215) or
           (item == 20052009) or (item == 20) or
           (item == 4965) or (item == 1691)):
            temp[i] = np.nan

    # state
    temp = data['state'].values
    for i, item in enumerate(temp):
        if item == 33:
            temp[i] = np.nan

    # material
    temp = data['material'].values
    for i, item in enumerate(temp):
        if item == 1:
            temp[i] = 9
        elif item == 2:
            temp[i] = 12
        elif item == 3:
            temp[i] = 7
        elif item == 4:
            temp[i] = 10
        elif item == 5:
            temp[i] = 11
        elif item == 6:
            temp[i] = 8
        elif np.isnan(item):
            temp[i] = 9.5

    # create house_age
    data['house_age'] = data['year'] - data['build_year']

    # floor, max_floor cleaning
    # remove weird floor, max_floor values
    temp = data['floor'].values
    temp2 = data['max_floor'].values
    for i, item in enumerate(temp):
        if temp2[i] == 0:
            temp2[i] = np.nan
        if item > temp2[i]:
            temp2[i] = np.nan
        elif item == 0:
            temp[i] = np.nan

    # subarea regional features

    data['population_raion'] = data['young_all'] + data['work_all'] + data['ekder_all']

    data['young_ratio_raion'] = data['young_all'] / data['population_raion']
    data['work_ratio_raion'] = data['work_all'] / data['population_raion']
    data['elder_ratio_raion'] = data['ekder_all'] / data['population_raion']

    data['population_density_raion'] = data['population_raion'] / data['area_m']

    data['preschool_quota_ratio'] = data['children_preschool'] / data['preschool_quota']
    data['school_quota_ratio'] = data['children_school'] / data['school_quota']

    data['school_education_centers_density'] = data['school_education_centers_raion'] / data['area_m']
    data['healthcare_centers_density'] = data['healthcare_centers_raion'] / data['area_m']
    data['sport_objects_density'] = data['sport_objects_raion'] / data['area_m']

    data['price_per_sqm'] = data['price_doc'] / data['full_sq']

    # lexical encoding
    categorical_list = list(data.select_dtypes(include=['object']).columns)
    for column in categorical_list:
        data[column] = pd.factorize(data[column].values, sort=True)[0]

    # output data with selected features
    main_features = [
        'id', 'price_doc', 'full_sq', 'life_sq', 'kitch_sq', 'floor', 'house_age',
        'max_floor', 'material', 'build_year', 'num_room', 'state', 'product_type']

    regional_features = [
        'metro_min_avto', 'metro_km_avto', 'metro_min_walk', 'metro_km_walk', 'kindergarten_km',
        'school_km', 'park_km', 'green_zone_km', 'industrial_km', 'water_treatment_km', 'cemetery_km',
        'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min',
        'railroad_station_avto_km', 'railroad_station_avto_min', 'public_transport_station_km',
        'public_transport_station_min_walk', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km',
        'bulvar_ring_km', 'kremlin_km', 'big_road1_km', 'big_road2_km', 'railroad_km', 'zd_vokzaly_avto_km',
        'bus_terminal_avto_km', 'oil_chemistry_km', 'nuclear_reactor_km', 'radiation_km',
        'power_transmission_line_km', 'thermal_power_plant_km', 'ts_km', 'big_market_km', 'market_shop_km',
        'fitness_km', 'swim_pool_km', 'ice_rink_km', 'stadium_km', 'basketball_km', 'hospice_morgue_km',
        'detention_facility_km', 'public_healthcare_km', 'university_km', 'workplaces_km',
        'shopping_centers_km', 'office_km', 'additional_education_km', 'preschool_km', 'big_church_km',
        'church_synagogue_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km'
    ]

    shopping_n_dining = [
        'trc_count_500', 'trc_count_1000', 'trc_count_1500', 'trc_count_2000', 'trc_count_3000', 'trc_count_5000',
        'cafe_count_500', 'cafe_count_1000', 'cafe_count_1500', 'cafe_count_2000', 'cafe_count_3000',
        'cafe_count_5000']

    office = [
        'office_count_500', 'office_count_1000', 'office_count_1500', 'office_count_2000', 'office_count_3000',
        'office_count_5000']

    sub_area_features = [
        'population_raion', 'young_ratio_raion', 'work_ratio_raion', 'elder_ratio_raion',
        'population_density_raion', 'preschool_quota_ratio', 'school_quota_ratio',
        'school_education_centers_density', 'healthcare_centers_density', 'sport_objects_density'
    ]

    surroundings = [
        'big_church_count_500', 'church_count_500', 'mosque_count_500', 'leisure_count_500', 'sport_count_500',
        'market_count_500', 'big_church_count_1000', 'church_count_1000', 'mosque_count_1000',
        'leisure_count_1000', 'sport_count_1000', 'market_count_1000', 'big_church_count_1500',
        'church_count_1500', 'mosque_count_1500', 'leisure_count_1500', 'sport_count_1500',
        'market_count_1500', 'big_church_count_2000', 'church_count_2000', 'mosque_count_2000',
        'leisure_count_2000', 'sport_count_2000', 'market_count_2000', 'big_church_count_3000',
        'church_count_3000', 'mosque_count_3000', 'leisure_count_3000', 'sport_count_3000',
        'market_count_3000', 'big_church_count_5000', 'church_count_5000', 'mosque_count_5000',
        'leisure_count_5000', 'sport_count_5000', 'market_count_5000']

    cafe_prices = [
        'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500',
        'cafe_sum_1000_min_price_avg', 'cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000',
        'cafe_sum_1500_min_price_avg', 'cafe_sum_1500_max_price_avg', 'cafe_avg_price_1500',
        'cafe_sum_2000_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000',
        'cafe_sum_3000_min_price_avg', 'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000',
        'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000']

    data[data['label'] == 1][main_features + regional_features +
                             sub_area_features + shopping_n_dining +
                             office + surroundings + cafe_prices
                             ].to_csv("preprocessed_training_data.csv", index=False)

    data[data['label'] == 0][main_features + regional_features +
                             sub_area_features + shopping_n_dining +
                             office + surroundings + cafe_prices
                             ].to_csv("preprocessed_testing_data.csv", index=False)

    print("Data preprocessed.")

if __name__ == "__main__":
    main()
