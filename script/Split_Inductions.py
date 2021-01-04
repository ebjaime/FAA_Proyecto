#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 18:54:37 2021

@author: marcos
"""
import numpy as np
import pylab as pl


metadata_np = np.loadtxt('../data/HT_Sensor_metadata.dat', skiprows=1, dtype=str)
metadata = 
dataset_np = np.loadtxt('../data/HT_Sensor_dataset.dat', skiprows=1)

# Expected: 36 wine, 33 banana, 31 background

# wine_ids = metadata