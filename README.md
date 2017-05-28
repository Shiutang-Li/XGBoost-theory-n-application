# XGBoost-theory-n-application

## Introduction: Theory

An introductory lecture to XGBoost is scheduled in [Big Data Utah meetup](https://www.meetup.com/BigDataUtah/events/238610160/). The lecture file ([talk.pdf](https://github.com/Shiutang-Li/Intro-to-XGBoost/blob/master/talk.pdf)) contains three main parts:

1. Analysis of XGBoost algorithm, with math explained in detail.

2. Introduction to major XGBoost parameters and parameter tuning.

3. A quick example showing how to apply XGBoost to kaggle Allstate Claims Severity dataset.  
Demo jupyter notebook: [Demo.ipynb](https://github.com/Shiutang-Li/Intro-to-XGBoost/blob/master/Demo.ipynb)

|![](3.jpg) | ![](4.jpg)| 
|:---:|:---:|

## Introduction: Application

Given the house price data in Moscow from 2011-2015, the goal of this project is to predict the house prices in Moscow from year 2015-2016. This is a competition currently hosted by kaggle: https://www.kaggle.com/c/sberbank-russian-housing-market

This notebook is not copied from any other kernels in kaggles, and it currently scores top 3% in public leader board (solo team, name: STL) (https://www.kaggle.com/c/sberbank-russian-housing-market/leaderboard/public?asOf=2017-5-29).

|![](1.jpg) | ![](2.jpg)| 
|:---:|:---:|

## Usage

(Main parameters / some preprocessing steps are removed because this competition is still in progress)

**Step 1.** Download train.csv and test.csv from https://www.kaggle.com/c/sberbank-russian-housing-market/data

**Step 2.**
```
python preprocessing.py
python build_xgboost_model.py
python predict.py
```

## Dependencies 

* Python 3.5
* xgboost 0.6
* pandas, numpy
