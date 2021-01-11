#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

par_dir = ".." # Por defecto se corre en directorio "script/"

columns_metadata = ["id", "date", "class", "t0", "dt"]
columns_dataset = ["id","time", "R1","R2","R3","R4","R5","R6","R7","R8", "Temp.", "Humidity"]

def load_metadata(columns_metadata=columns_metadata, par_dir=par_dir):
    metadata = np.loadtxt(par_dir+'/data/HT_Sensor_metadata.dat', skiprows=1, dtype=str)
    metadata_df = pd.DataFrame(data=metadata, columns=columns_metadata)
    metadata_df = metadata_df.astype({'id': int, 't0':float, 'dt':float}) 
    return metadata_df

def load_dataset(columns_dataset=columns_dataset, par_dir=par_dir):
    dataset = np.loadtxt(par_dir+'/data/HT_Sensor_dataset.dat', skiprows=1)
    dataset_df = pd.DataFrame(data=dataset, columns=columns_dataset)
    return dataset_df

# Grafica temperatura vs humedad media
def plotTempHum(poblacion, c):
    plt.scatter(poblacion.groupby("id")["Temp."].mean(), 
                poblacion.groupby("id")["Humidity"].mean(),
                c=c)
    plt.xlabel("Grados ºC")
    plt.ylabel("% Humidity")


# Separacion de ids por clase 
def split_class_id(metadata_df):
    dataset_ban = metadata_df[metadata_df["class"] == "banana"].id
    dataset_win = metadata_df[metadata_df["class"] == "wine"].id
    dataset_bac = metadata_df[metadata_df["class"] == "background"].id
    
    return dataset_ban, dataset_win, dataset_bac

# Dividir dataset en antes, durante y despues de experimentos
def separar_dataset_segun_tiempo(dataset_df, metadata_df):
    dataset_antes = pd.DataFrame(columns=dataset_df.columns)
    dataset_durante = pd.DataFrame(columns=dataset_df.columns)
    dataset_despues = pd.DataFrame(columns=dataset_df.columns)
    for id in dataset_df.id.unique():
        dataset_aux = dataset_df[dataset_df.id==id].copy() # Obtengo copia de sub-dataset con id utilizado
        # Obtengo informacion sobre los tiempos
        info_t = metadata_df[metadata_df.id == id].values[0]
        t0 = info_t[3] # Empieza simulacion
        tf = t0 + info_t[4] #Finaliza simulacion
        dataset_aux.time += t0
        # Separo los datos y aniado a dataframes auxiliares
        dataset_antes=dataset_antes.append(dataset_aux[dataset_aux.time < t0])
        dataset_durante=dataset_durante.append(dataset_aux[(dataset_aux.time >= t0) & (dataset_aux.time <= tf)])
        dataset_despues=dataset_despues.append(dataset_aux[dataset_aux.time > tf])
        
    return dataset_antes, dataset_durante, dataset_despues


def get_missing_values(metadata_df, dataset_df=None, dfs=None):
    diffs = []
    if dfs is None:
        if dataset_df is None:
            print("Falta el valor de metadata. Sintaxis: metadata=df, dataset=df, dfs=[]")
            return -1
        dataset_antes, dataset_durante, dataset_despues = separar_dataset_segun_tiempo(dataset_df, metadata_df)
        dfs = [dataset_antes, dataset_durante, dataset_despues]
    
    for df in dfs:
        diffs.append(metadata_df.loc[ metadata_df["id"].isin(df["id"]) != True].id)
        
    return diffs

# ???
def get_class(metadata, clases=None):
    if clases is None:
        clases = metadata["class"]
    clases1 = clases.drop([95])
    clases2 = clases.drop([14, 76, 95])
    return clases1, clases2

def get_class_lists(metadata):
    clases = {}
    temp = metadata["class"]
    clases["all"] = list(get_class(metadata, temp.map({"banana": 0, "wine": 1, "background": 2})))
    clases["banana"] = list(get_class(metadata, temp.map({"banana": 1, "wine": 0, "background": 0}) ))
    clases["wine"] = list(get_class(metadata, temp.map({"banana": 0, "wine": 1, "background": 0})))
    clases["background"] = list(get_class(metadata, temp.map({"banana": 0, "wine": 0, "background": 1})))
    
    return clases
    

def get_split_np_data(dataset_df, metadata_df):
    vars_dataset_antes, vars_dataset_durante, vars_dataset_despues = vars_dataset_segun_tiempo(dataset_df, metadata_df)
    return (np.array(list(vars_dataset_antes.values())), 
            np.array(list(vars_dataset_durante.values())), 
            np.array(list(vars_dataset_despues.values())))


# Varianza para cada sensor respecto a un mismo experimento de dataset
def varianzas_sensores(dataset_df):
    vars_dataset = {}
    for id in dataset_df.id.unique():
        aux=[]
        for sensor in dataset_df.columns[2:10]:
            aux.append(dataset_df[dataset_df.id==id][sensor].var())
        vars_dataset[id]=aux
    return vars_dataset

# Media para cada sensor respecto a un mismo experimento de dataset
def medias_sensores(dataset_df):
    means_dataset = {}
    for id in dataset_df.id.unique():
        aux=[]
        for sensor in dataset_df.columns[2:10]:
            aux.append(dataset_df[dataset_df.id==id][sensor].mean())
        means_dataset[id]=aux
    return means_dataset

# Media para cada sensor respecto a un mismo experimento de dataset
def max_sensores(dataset_df):
    max_dataset = {}
    for id in dataset_df.id.unique():
        aux=[]
        for sensor in dataset_df.columns[2:10]:
            aux.append(dataset_df[dataset_df.id==id][sensor].max())
        max_dataset[id]=aux
    return max_dataset

# Media para cada sensor respecto a un mismo experimento de dataset
def min_sensores(dataset_df):
    min_dataset = {}
    for id in dataset_df.id.unique():
        aux=[]
        for sensor in dataset_df.columns[2:10]:
            aux.append(dataset_df[dataset_df.id==id][sensor].min())
        min_dataset[id]=aux
    return min_dataset

# Calcula medias de varianzas o medias de sensores de cada experimento de dataset
def mean_varmean_sensores(dataset_df, var=True, mean=False, processed_dataset=None):
    if processed_dataset is None:
        if var is True:
            processed_dataset = varianzas_sensores(dataset_df)
        if mean is False:
            processed_dataset = medias_sensores(dataset_df)
    mean_proc_dataset = {}
    for id in processed_dataset.keys():
        mean_proc_dataset[id] = np.mean(processed_dataset[id])
    
    return mean_proc_dataset

# Varianzas de cada etapa agrupado por experimento
def vars_dataset_segun_tiempo(dataset_df, metadata_df):
    dataset_antes, dataset_durante, dataset_despues = separar_dataset_segun_tiempo(dataset_df, metadata_df)
    vars_dataset_antes = varianzas_sensores(dataset_antes)
    vars_dataset_durante = varianzas_sensores(dataset_durante)
    vars_dataset_despues = varianzas_sensores(dataset_despues)
    return vars_dataset_antes, vars_dataset_durante, vars_dataset_despues

# Medias de varianzas anteriores
def means_vars_dataset_segun_tiempo(vars_dataset_antes, vars_dataset_durante, vars_dataset_despues):
    
    mean_vars_dataset_antes = mean_varmean_sensores(processed_dataset = vars_dataset_antes)
    mean_vars_dataset_durante = mean_varmean_sensores(processed_dataset = vars_dataset_durante)
    mean_vars_dataset_despues = mean_varmean_sensores(processed_dataset = vars_dataset_despues)
    
    return mean_vars_dataset_antes, mean_vars_dataset_durante, mean_vars_dataset_despues
