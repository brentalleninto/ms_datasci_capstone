# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:40:31 2018

@author: brent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

quaketest = pd.read_csv('test_values.csv')
quaketrain = pd.read_csv('train_values.csv')
quaketrainlabels = pd.read_csv('train_labels.csv')
#print(quaketrain.describe())
#quaketrainlabels['damage_grade'].hist()

quaketrainlabeled = quaketrain.merge(quaketrainlabels, left_on='building_id', right_on='building_id')

