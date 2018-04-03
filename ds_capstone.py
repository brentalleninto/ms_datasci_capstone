# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:40:31 2018

@author: brent
"""

# Importing required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer


# Importing training and test sets.
X_holdout = pd.read_csv('test_values.csv')
X = pd.read_csv('train_values.csv')
y = pd.read_csv('train_labels.csv')
all_train = X.merge(y, left_on='building_id', right_on='building_id')


# Setup columns as accurate data types.
categorical_columns = ['geo_level_1_id','geo_level_2_id','geo_level_3_id',
                       'land_surface_condition','foundation_type','roof_type',
                       'ground_floor_type','other_floor_type','position',
                       'plan_configuration', 'legal_ownership_status']

miscoded_integers = ['count_families','has_secondary_use']
non_binary_integers = ['area']
columns_to_drop = ['building_id','geo_level_2_id','geo_level_3_id']

#sns.factorplot(y='geo_level_1_id',hue='damage_grade',kind='count',data=all_train)

#Converting categorical columns to categorical.
for col in categorical_columns:
    all_train[col] = all_train[col].astype('category')
    X_holdout[col] = X_holdout[col].astype('category')

#Dropping columns which do not have predictive value.
for col in columns_to_drop:
    all_train = all_train.drop([col], axis=1)
    X_holdout = X_holdout.drop([col], axis=1)

#Scaling non-binary integers.
for col in non_binary_integers:
    all_train[col] = stats.boxcox(all_train[col])[0]
    X_holdout[col] = stats.boxcox(X_holdout[col])[0]

#pd.get_dummies(all_train)
all_train = pd.get_dummies(all_train)
X_holdout = pd.get_dummies(X_holdout)

for col in miscoded_integers:
    all_train[col] = all_train[col].astype('int64')
    X_holdout[col] = X_holdout[col].astype('int64')

#Dropping outliers.

X = all_train.drop(['damage_grade'], axis=1)
y = all_train['damage_grade']

#EDA
#sns.factorplot(y='age',hue='damage_grade',kind='count',data=all_train)
#sns.factorplot(y='count_floors_pre_eq',hue='damage_grade',kind='count',data=all_train)

#This classifier was used to determine feature values.

gb_params = {'learning_rate':[0.15],
             'min_samples_split':[8]
             }

cls_model = GradientBoostingClassifier()
clf = GridSearchCV(cls_model, param_grid=gb_params, scoring='f1_micro', cv=3)
#clf = RandomizedSearchCV(cls_model, param_distributions=gb_params, scoring='f1_micro', cv=3)
clf.fit(X, y)
print("Cross-validated score: {}".format(clf.best_score_))
print("Best parameters {}".format(clf.best_params_))

#y_pred = clf.predict(X_holdout)

'''
cls_model.fit(X, y)
importance = cls_model.feature_importances_
importance_graph = pd.Series(importance, index=X.columns)
importance_graph = importance_graph.nlargest(100)[importance_graph > 0.01]
importance_graph.plot(kind='barh')
'''