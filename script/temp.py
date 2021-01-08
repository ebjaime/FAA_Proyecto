#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:34:23 2021

@author: marcos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab as pl
from Plot_Induction_Figure import graficaInduccion 

prev_dir = "../"

columns_metadata = ["id", "date", "class", "t0", "dt"]
metadata = np.loadtxt(prev_dir+'data/HT_Sensor_metadata.dat', skiprows=1, dtype=str)
# sensor_m = pd.read_csv(prev_dir+"data/HT_Sensor_metadata.dat", sep="\t")
metadata_df = pd.DataFrame(data=metadata, columns=columns_metadata)

columns_dataset = ["id","time", "R1","R2","R3","R4","R5","R6","R7","R8", "Temp.", "Humidity"]
dataset = np.loadtxt(prev_dir+'data/HT_Sensor_dataset.dat', skiprows=1)
dataset_df = pd.DataFrame(data=dataset, columns=columns_dataset)


# Separacion de ids por clase 
dataset_ban = metadata_df[metadata_df["class"] == "banana"].id
dataset_win = metadata_df[metadata_df["class"] == "wine"].id
dataset_bac = metadata_df[metadata_df["class"] == "background"].id



# Varianza para cada sensor respecto a un mismo experimento de dataset
def varianzas_sensores(dataset=dataset):
    vars_dataset = {}
    for id in dataset.id.unique():
        aux=[]
        for sensor in dataset.columns[2:10]:
            aux.append(dataset[dataset.id==id][sensor].var())
        vars_dataset[id]=aux
    return vars_dataset

# Media para cada sensor respecto a un mismo experimento de dataset
def medias_sensores(dataset=dataset):
    means_dataset = {}
    for id in dataset.id.unique():
        aux=[]
        for sensor in dataset.columns[2:10]:
            aux.append(dataset[dataset.id==id][sensor].mean())
        means_dataset[id]=aux
    return means_dataset


# Calcula medias de varianzas o medias de sensores de cada experimento de dataset
def mean_varmean_sensores(dataset=dataset, var=True, mean=False, processed_dataset=None):
    if processed_dataset is None:
        if var is True:
            processed_dataset = varianzas_sensores(dataset)
        if mean is False:
            processed_dataset = medias_sensores(dataset)
    mean_proc_dataset = {}
    for id in processed_dataset.keys():
        mean_proc_dataset[id] = np.mean(processed_dataset[id])
    
    return mean_proc_dataset

# Dividir dataset en antes, durante y despues de experimentos
def separar_dataset_segun_tiempo(dataset, metadata):
    dataset_antes = pd.DataFrame(columns=dataset.columns)
    dataset_durante = pd.DataFrame(columns=dataset.columns)
    dataset_despues = pd.DataFrame(columns=dataset.columns)
    for id in dataset.id.unique():
        dataset_aux = dataset[dataset.id==id].copy() # Obtengo copia de sub-dataset con id utilizado
        # Obtengo informacion sobre los tiempos
        info_t = metadata[metadata.id == id].values[0]
        t0 = info_t[3] # Empieza simulacion
        tf = t0 + info_t[4] #Finaliza simulacion
        dataset_aux.time += t0
        # Separo los datos y aniado a dataframes auxiliares
        dataset_antes=dataset_antes.append(dataset_aux[dataset_aux.time < t0])
        dataset_durante=dataset_durante.append(dataset_aux[(dataset_aux.time >= t0) & (dataset_aux.time <= tf)])
        dataset_despues=dataset_despues.append(dataset_aux[dataset_aux.time > tf])
        
    return dataset_antes, dataset_durante, dataset_despues


dataset_antes, dataset_durante, dataset_despues = separar_dataset_segun_tiempo(sensor_d, sensor_m)


# Varianzas de cada etapa agrupado por experimento
vars_dataset_antes = varianzas_sensores(dataset_antes)
vars_dataset_durante = varianzas_sensores(dataset_durante)
vars_dataset_despues = varianzas_sensores(dataset_despues)

# Medias de varianzas anteriores
mean_vars_dataset_antes = mean_varmean_sensores(processed_dataset = vars_dataset_antes)
mean_vars_dataset_durante = mean_varmean_sensores(processed_dataset = vars_dataset_durante)
mean_vars_dataset_despues = mean_varmean_sensores(processed_dataset = vars_dataset_despues)
