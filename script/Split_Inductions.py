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
# from Plot_Induction_Figure import graficaInduccion 

prev_dir = "../"

columns_metadata = ["id", "date", "class", "t0", "dt"]
columns_dataset = ["id","time", "R1","R2","R3","R4","R5","R6","R7","R8", "Temp.", "Humidity"]

metadata = np.loadtxt(prev_dir+'data/HT_Sensor_metadata.dat', skiprows=1, dtype=str)
# metadata_df = pd.DataFrame(data=metadata, columns=columns_metadata)
metadata_df = pd.read_csv(prev_dir+"data/HT_Sensor_metadata.dat", sep="\t")
# Correccion: cabecera de metadata mal leido
metadata_df.drop(["dt"], axis=1, inplace=True)
metadata_df.rename(columns={"t0":"dt","class":"t0", "Unnamed: 2":"class"}, inplace=True)

dataset = np.loadtxt(prev_dir+'data/HT_Sensor_dataset.dat', skiprows=1)
# dataset_df = pd.DataFrame(data=dataset, columns=columns_dataset)
dataset_df = pd.read_csv(prev_dir+"data/HT_Sensor_dataset.dat", sep="  ", engine="python")
# Correccion: datos de dataset mal leidos
dataset_df.dropna(axis=1, inplace=True)
dataset_df.columns=["id","time", "R1","R2","R3","R4","R5","R6","R7","R8", "Temp.", "Humidity"]


def load_metadata():
    metadata = np.loadtxt(prev_dir+'data/HT_Sensor_metadata.dat', skiprows=1, dtype=str)
    metadata_df = pd.DataFrame(data=metadata, columns=columns_metadata)
    return metadata, metadata_df

def load_dataset():
    dataset = np.loadtxt(prev_dir+'data/HT_Sensor_dataset.dat', skiprows=1)
    dataset_df = pd.DataFrame(data=dataset, columns=columns_dataset)
    return dataset, dataset_df

# Separacion de ids por clase 
def split_class_id(metadata_df):
    dataset_ban = metadata_df[metadata_df["class"] == "banana"].id
    dataset_win = metadata_df[metadata_df["class"] == "wine"].id
    dataset_bac = metadata_df[metadata_df["class"] == "background"].id
    
    return dataset_ban, dataset_win, dataset_bac

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
def separar_dataset_segun_tiempo(dataset=dataset_df, metadata=metadata_df):
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

# Varianzas de cada etapa agrupado por experimento
def vars_dataset_segun_tiempo(dataset=dataset_df, metadata=metadata_df):
    dataset_antes, dataset_durante, dataset_despues = separar_dataset_segun_tiempo()
    vars_dataset_antes = varianzas_sensores(dataset_antes)
    vars_dataset_durante = varianzas_sensores(dataset_durante)
    vars_dataset_despues = varianzas_sensores(dataset_despues)
    return vars_dataset_antes, vars_dataset_durante, vars_dataset_despues

# Medias de varianzas anteriores
def means_vars_dataset_segun_tiempo(dataset=dataset_df, vars_dataset_antes=None, vars_dataset_durante=None, vars_dataset_despues=None, metadata=metadata_df):
    if metadata is not None:
        dataset_antes, dataset_durante, dataset_despues = separar_dataset_segun_tiempo()
        
    mean_vars_dataset_antes = mean_varmean_sensores(processed_dataset = vars_dataset_antes)
    mean_vars_dataset_durante = mean_varmean_sensores(processed_dataset = vars_dataset_durante)
    mean_vars_dataset_despues = mean_varmean_sensores(processed_dataset = vars_dataset_despues)
    
    return mean_vars_dataset_antes, mean_vars_dataset_durante, mean_vars_dataset_despues


def get_missing_values(dataset=dataset_df, dfs=None, metadata=None):
    diffs = []
    if dfs is None:
        if metadata is None:
            print("Falta el valor de metadata. Sintaxis: dataset=df, dfs=[], metadata=df")
            return -1
        dataset_antes, dataset_durante, dataset_despues = separar_dataset_segun_tiempo(dataset, metadata)
        dfs = [dataset_antes, dataset_durante, dataset_despues]
    
    for df in dfs:
        diffs.append(metadata.loc[ metadata["id"].isin(df["id"]) != True])
        
    return diffs

def get_class_lists(metadata=metadata):
    clases = np.array( metadata[:,[2]] )
    clases1 = np.delete(clases, 95, 0).ravel()
    clases2 = np.delete(clases, [14, 76, 95], 0).ravel()
    
    return clases1, clases2

def get_split_np_data(dataset=dataset_df, metadata=metadata_df):
    vars_dataset_antes, vars_dataset_durante, vars_dataset_despues = vars_dataset_segun_tiempo()
    return np.array(list(vars_dataset_antes.values())), np.array(list(vars_dataset_durante.values())), np.array(list(vars_dataset_despues.values()))