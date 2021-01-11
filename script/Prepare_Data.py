#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:45:12 2021

@author: marcos
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from Preprocess import *

titles = ["Variaciones durante la induccion", "Variaciones antes y durante la induccion",
          "Variaciones durante y despues de la induccion",
          "Variaciones antes, durante y despues de la induccion",
          "Diferencia de Variaciones antes y durante la induccion",
          "Diferencia de Variaciones durante y despues de la induccion",
          "Diferencia de Variaciones antes, durante y despues de la induccion"]

def merge_np_data(data1, data2, delete1=False, delete2=False):
    if delete1:
        data1 = np.delete(data1, [14, 76], 0)
    if delete2:
        data2 = np.delete(data2, [14, 76], 0)
    return np.column_stack((data1, data2))

def substraction(data1, data2, delete1=False, delete2=False):
    if delete1:
        data1 = np.delete(data1, [14, 76], 0)
    if delete2:
        data2 = np.delete(data2, [14, 76], 0)
    return np.absolute(np.subtract(data1, data2))


        
def get_data_models(metadata=None, dataset=None, clases1=None, clases2=None):
    conj_datos = []
    if metadata is None:
        metadata = load_metadata()
    if dataset is None:
        dataset = load_dataset()
    vars1, vars2, vars3 = get_split_np_data(dataset, metadata)
    if clases1 is None or clases2 is None:
        clases1, clases2 = get_class(metadata)
    
    # Guardamos el conjunto de datos de prueba las varianzas de antes de la induccion
    # conj_datos.append({"X": vars1, "y": clases1}) 
    # Guardamos el conjunto de datos de prueba las varianzas de durante la induccion
    conj_datos.append({"X": vars2, "y": clases1})
    # Guardamos el conjunto de datos de prueba las varianzas de despues de la induccion
    # conj_datos.append({"X": vars3, "y": clases2})
    
    # Guardamos el conjunto de datos de prueba las varianzas de antes y durante la induccion
    vars4 = merge_np_data(vars1, vars2)
    conj_datos.append({"X": vars4, "y": clases1})
    # Guardamos el conjunto de datos de prueba las varianzas de durante y despues de la induccion
    vars5 = merge_np_data(vars2, vars3, delete1=True)
    conj_datos.append({"X": vars5, "y": clases2})
    # Guardamos el conjunto de datos de prueba las varianzas de antes, durante y despues de la induccion
    vars6 = merge_np_data(vars1, vars5, delete1=True)
    conj_datos.append({"X": vars6, "y": clases2})
    
    # Guardamos el conjunto de datos de prueba la diferencia de varianzas de antes y durante induccion
    vars7 = substraction(vars1, vars2)
    conj_datos.append({"X": vars7, "y": clases1})
    # Guardamos el conjunto de datos de prueba la diferencia de varianzas de durante y despues de la induccion
    vars8 = substraction(vars2, vars3, delete1=True)
    conj_datos.append({"X": vars8, "y": clases2})
    # Guardamos el conjunto de datos de prueba la diferencia de varianzas de antes, durante y despues de la induccion
    vars9 = merge_np_data(vars7, vars8, delete1=True)
    conj_datos.append({"X": vars9, "y": clases2})
    
    return conj_datos

def validations(conj_datos=None):
    scores = []
    if conj_datos is None:
        conj_datos = get_data_models()
    
    for idx, clf in enumerate(clfs):
        print("\n\nPruebas con clasificador: ", clfs_name[idx])
        for i, c in enumerate(conj_datos):
            print("\nScores para el conjunto de datos: ", titles[i])
            scores.append(cross_val_score(clf, c["X"], c["y"], cv=5))
            print("\t",scores[i])
            print("\tAcierto medio: ", np.round(scores[idx*len(clfs)+i].mean(), 4))
    
    return scores

metadata = load_metadata()
dataset = load_dataset()
clases = get_class_lists(metadata)


clfs = []
# clfs.append(svm.SVC(kernel='linear', C=1, random_state=0))
# clfs.append(LogisticRegression(random_state=0, max_iter=1000))
# clfs.append(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0))
clfs.append(RandomForestClassifier(max_depth=2, random_state=0))
clfs_name = [ 
    # "SVM",
    # "Logistic Regresion", 
    # "Neuronal Network",
    "Random Forest"
    ]

for c in clases:
    print("PRUEBAS CON LA CLASE DE TIPO:", c)
    conj_datos = get_data_models(metadata, dataset, clases1=clases[c][0], clases2=clases[c][1] )
    scores = validations(conj_datos)  
    print("\n\n\n")



scores = validations(conj_datos)  