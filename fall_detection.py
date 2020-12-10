#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:56:25 2020

@author: arnautienda
"""
from os import walk
from art import *
import pandas as pd
import os
import pickle
import math
import sys
import numpy as np


print("---------------------------------")
tprint("FALL")
tprint("DETECTION")
print("---------------------------------")

print("Instructions:")

print("\tFile accepted format: 3 columns, separated by commas, one for each axis (X,Y,Z).")
print("\tValues need to be coded in g (1g = 9.81m/s^2).")
print("\nExample:\n")

print("\        X      Y      Z") 
print("4  -0.998 -0.270  0.151")
print("5  -0.998 -0.270  0.151")
print("6  -1.000 -0.332  0.192")
print("7  -0.976 -0.373  0.133")
print("8  -0.976 -0.373  0.133")
print("..    ...    ...    ...")


print("---------------------------------\n")
path = input("Introduce the path to your csv file: ")
sampling_freq = 'a'
while not sampling_freq.isdigit():
    sampling_freq = input("Sampling frequency of your recording (in Hz): ")
    if not sampling_freq.isdigit():
        print("\nIntroduce a valid sampling frequency.")
trial_file_name_aux = os.path.basename(path)
trial_file_name = os.path.splitext(trial_file_name_aux)[0]


def compute_file(path, sampling_freq, trial_file_name):    
    ok_file = 0
    # Create a dataframe from the csv file  
    try:
        df_Mediciones = pd.DataFrame(pd.read_csv(path, header = None, sep = ',', 
    											 names = ["X", "Y", "Z"], 
    													 skiprows= 1))
        ok_file = 1
    except:
        ok_file = 0
    if ok_file == 1:    
        sf = int(sampling_freq)
        ws = 3 # window size in seconds
        ws_samples = sf * ws # window size in samples
        
        
        # Calculate the Euclidean Norm of (XYZ) Acceleration
        fn = lambda row: math.sqrt((row.X)**2 + (row.Y)**2 + (row.Z)**2) 
        col = df_Mediciones.apply(fn, axis=1) 
        df_Mediciones = df_Mediciones.assign(N_XYZ=col.values)   
        
        
        # Get the max value of N_XYZ
        # This max value is used as a reference point to get a window of values
        max_N_XYZ = np.max(df_Mediciones.N_XYZ)
        
        max_N = np.max(df_Mediciones.N_XYZ)
        max_N_index = df_Mediciones.index[df_Mediciones.N_XYZ == max_N][0]
        min_N = np.min(df_Mediciones.N_XYZ)
        min_N_index = df_Mediciones.index[df_Mediciones.N_XYZ == min_N][0]
        len_df_Mediciones = len(df_Mediciones)
        
        # We work with a window of 600 measurements (equivalent to 3 seconds of activity recording)
        # We handle three scenarios, the window falls completely within the values, or the windows is aligned
        # with the minimum/maximum value...
        if (max_N_index - round(ws_samples/2)<0):
            df_Mediciones = df_Mediciones[0:round(ws_samples+1)]
        else:
            if (max_N_index + round(ws_samples/2)+1> len_df_Mediciones):
                df_Mediciones = df_Mediciones[len_df_Mediciones-round(ws_samples+2):len_df_Mediciones-1]
            else:
                # extract the central window
                df_Mediciones = df_Mediciones[max_N_index - round(ws_samples/2): max_N_index + round(ws_samples/2)+1]
               
        # N_HOR: Calculate the Euclidean Norm of (HOR) Acceleration
        fn_hor = lambda row: math.sqrt((row.Y)**2 + (row.Z)**2) # define a function for the new column
        col = df_Mediciones.apply(fn_hor, axis=1) # get column data with an index
        df_Mediciones = df_Mediciones.assign(N_HOR=col.values) # assign values to column 'c'  
        
        # N_VER: Calculate the Euclidean Norm of (VER) Acceleration
        fn_ver = lambda row: math.sqrt((row.X)**2 + (row.Z)**2) # define a function for the new column
        col = df_Mediciones.apply(fn_ver, axis=1) # get column data with an index
        df_Mediciones = df_Mediciones.assign(N_VER=col.values) # assign values to column 'c' 
        
        # Calculate multiple characteristics of X axis    
        field_name = "X"
        
        var_X = df_Mediciones[field_name].var()
        mean_X = df_Mediciones[field_name].mean()
        std_X = df_Mediciones[field_name].std()
        median_X = df_Mediciones[field_name].median()
        max_X = df_Mediciones[field_name].max()
        min_X = df_Mediciones[field_name].min()
        range_X = max_X - min_X
        kurtosis_X = df_Mediciones[field_name].kurtosis()
        skewness_X =  df_Mediciones[field_name].skew()
        
        df_Features = pd.DataFrame({"var_" + field_name: [var_X], 
                                    "mean_" + field_name:[mean_X],
                                    "std_" + field_name:[std_X], 
                                    "max_" + field_name:[max_X],
                                    "min_" + field_name:[min_X], 
                                    "range_" + field_name:[range_X],
                                    "kurtosis_" + field_name:[kurtosis_X],
                                    "skewness_" + field_name:[skewness_X]})        
                          
        # Calculate multiple characteristics of Y axis
        field_name = "Y"
        
        var_X = df_Mediciones[field_name].var()
        mean_X = df_Mediciones[field_name].mean()
        std_X = df_Mediciones[field_name].std()
        median_X = df_Mediciones[field_name].median()
        max_X = df_Mediciones[field_name].max()
        min_X = df_Mediciones[field_name].min()
        range_X = max_X - min_X
        kurtosis_X = df_Mediciones[field_name].kurtosis()
        skewness_X =  df_Mediciones[field_name].skew()
        
        df_Features_2 = pd.DataFrame({"var_" + field_name: [var_X], 
                                    "mean_" + field_name:[mean_X],
                                    "std_" + field_name:[std_X], 
                                    "max_" + field_name:[max_X],
                                    "min_" + field_name:[min_X], 
                                    "range_" + field_name:[range_X],
                                    "kurtosis_" + field_name:[kurtosis_X],
                                    "skewness_" + field_name:[skewness_X]})                              
        
        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        # Calculate multiple characteristics of Z axis
        field_name = "Z"
        
        var_X = df_Mediciones[field_name].var()
        mean_X = df_Mediciones[field_name].mean()
        std_X = df_Mediciones[field_name].std()
        median_X = df_Mediciones[field_name].median()
        max_X = df_Mediciones[field_name].max()
        min_X = df_Mediciones[field_name].min()
        range_X = max_X - min_X
        kurtosis_X = df_Mediciones[field_name].kurtosis()
        skewness_X =  df_Mediciones[field_name].skew()    
        
        df_Features_2 = pd.DataFrame({"var_" + field_name: [var_X], 
                                    "mean_" + field_name:[mean_X],
                                    "std_" + field_name:[std_X], 
                                    "max_" + field_name:[max_X],
                                    "min_" + field_name:[min_X], 
                                    "range_" + field_name:[range_X],
                                    "kurtosis_" + field_name:[kurtosis_X],
                                    "skewness_" + field_name:[skewness_X]})                              
        
        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        # acceleration features
        # Calculate multiple characteristics of Euclidean Norm Total
        field_name = "N_XYZ"
        
        var_X = df_Mediciones[field_name].var()
        mean_X = df_Mediciones[field_name].mean()
        std_X = df_Mediciones[field_name].std()
        median_X = df_Mediciones[field_name].median()
        max_X = df_Mediciones[field_name].max()
        min_X = df_Mediciones[field_name].min()
        range_X = max_X - min_X
        kurtosis_X = df_Mediciones[field_name].kurtosis()
        skewness_X =  df_Mediciones[field_name].skew()    
        
        df_Features_2 = pd.DataFrame({"var_" + field_name: [var_X], 
                                    "mean_" + field_name:[mean_X],
                                    "std_" + field_name:[std_X], 
                                    "max_" + field_name:[max_X],
                                    "min_" + field_name:[min_X], 
                                    "range_" + field_name:[range_X],
                                    "kurtosis_" + field_name:[kurtosis_X],
                                    "skewness_" + field_name:[skewness_X]})                              
        
        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        # acceleration features
        # Calculate multiple characteristics of Euclidean Norm Horizontal
        field_name = "N_HOR"
        
        var_X = df_Mediciones[field_name].var()
        mean_X = df_Mediciones[field_name].mean()
        std_X = df_Mediciones[field_name].std()
        median_X = df_Mediciones[field_name].median()
        max_X = df_Mediciones[field_name].max()
        min_X = df_Mediciones[field_name].min()
        range_X = max_X - min_X
        kurtosis_X = df_Mediciones[field_name].kurtosis()
        skewness_X =  df_Mediciones[field_name].skew()    
        
        df_Features_2 = pd.DataFrame({"var_" + field_name: [var_X], 
                                    "mean_" + field_name:[mean_X],
                                    "std_" + field_name:[std_X], 
                                    "max_" + field_name:[max_X],
                                    "min_" + field_name:[min_X], 
                                    "range_" + field_name:[range_X],
                                    "kurtosis_" + field_name:[kurtosis_X],
                                    "skewness_" + field_name:[skewness_X]})                              
        
        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        # Calculate multiple characteristics of Euclidean Norm Vertical
        field_name = "N_VER"
        
        var_X = df_Mediciones[field_name].var()
        mean_X = df_Mediciones[field_name].mean()
        std_X = df_Mediciones[field_name].std()
        median_X = df_Mediciones[field_name].median()
        max_X = df_Mediciones[field_name].max()
        min_X = df_Mediciones[field_name].min()
        range_X = max_X - min_X
        kurtosis_X = df_Mediciones[field_name].kurtosis()
        skewness_X =  df_Mediciones[field_name].skew()    
        
        df_Features_2 = pd.DataFrame({"var_" + field_name: [var_X], 
                                    "mean_" + field_name:[mean_X],
                                    "std_" + field_name:[std_X], 
                                    "max_" + field_name:[max_X],
                                    "min_" + field_name:[min_X], 
                                    "range_" + field_name:[range_X],
                                    "kurtosis_" + field_name:[kurtosis_X],
                                    "skewness_" + field_name:[skewness_X]})                              
        
        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        corr_valueXY = df_Mediciones["X"].corr(df_Mediciones["Y"])
        corr_valueXZ = df_Mediciones["X"].corr(df_Mediciones["Z"])
        corr_valueYZ = df_Mediciones["Y"].corr(df_Mediciones["Z"])
        corr_valueNV = df_Mediciones["N_XYZ"].corr(df_Mediciones["N_VER"])
        corr_valueNH = df_Mediciones["N_XYZ"].corr(df_Mediciones["N_HOR"])
        corr_valueHV = df_Mediciones["N_HOR"].corr(df_Mediciones["N_VER"])
        
        df_Features_2 = pd.DataFrame({"corr_XY": [corr_valueXY], 
                                    "corr_XZ": [corr_valueXZ],
                                    "corr_YZ": [corr_valueYZ],
                                    "corr_NV": [corr_valueNV],
                                    "corr_NH": [corr_valueNH], 
                                    "corr_HV": [corr_valueHV]})       
        
        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        df_Features_2 = pd.DataFrame({"File": [trial_file_name],
                                      "Fall_ADL": "X",
                                      "Act_Type": "X",
                                     })  
        
        df_Features =  pd.concat([df_Features_2, df_Features], axis=1)
        
    
        return df_Features
    
    return 0
    
def detection(df_Features):
    
    

    
    colnames = [" ",'File', 'Fall_ADL', 'Act_Type', 'var_X', 'mean_X',
           'std_X', 'max_X', 'min_X', 'range_X', 'kurtosis_X',
           'skewness_X', 'var_Y', 'mean_Y', 'std_Y', 'max_Y',
           'min_Y', 'range_Y', 'kurtosis_Y', 'skewness_Y', 'var_Z',
           'mean_Z', 'std_Z','max_Z', 'min_Z', 'range_Z',
           'kurtosis_Z', 'skewness_Z', 'var_N_XYZ', 'mean_N_XYZ',
           'std_N_XYZ', 'max_N_XYZ', 'min_N_XYZ', 'range_N_XYZ',
           'kurtosis_N_XYZ', 'skewness_N_XYZ', 'var_N_HOR',
           'mean_N_HOR', 'std_N_HOR', 'max_N_HOR', 'min_N_HOR',
           'range_N_HOR', 'kurtosis_N_HOR', 'skewness_N_HOR',
           'var_N_VER', 'mean_N_VER', 'std_N_VER', 'max_N_VER',
           'min_N_VER', 'range_N_VER', 'kurtosis_N_VER',
           'skewness_N_VER', 'corr_XY', 'corr_XZ', 'corr_YZ', 'corr_NV',
           'corr_NH', 'corr_HV']
    
    x_columns = ['kurtosis_X','max_X','mean_X','min_X','range_X','skewness_X','std_X','var_X',
             'kurtosis_Y','max_Y','mean_Y','min_Y','range_Y','skewness_Y','std_Y','var_Y',
             'kurtosis_Z','max_Z','mean_Z','min_Z','range_Z','skewness_Z','std_Z','var_Z',
             'kurtosis_N_XYZ','max_N_XYZ','mean_N_XYZ','min_N_XYZ','range_N_XYZ','skewness_N_XYZ','std_N_XYZ','var_N_XYZ',
             'kurtosis_N_HOR','max_N_HOR','mean_N_HOR','min_N_HOR','range_N_HOR','skewness_N_HOR','std_N_HOR','var_N_HOR',
             'kurtosis_N_VER','max_N_VER','mean_N_VER','min_N_VER','range_N_VER','skewness_N_VER','std_N_VER','var_N_VER',
             'corr_HV','corr_NH','corr_NV','corr_XY','corr_XZ','corr_YZ']

    model = open("RFT_model.pkl",'rb')
    rf_model = pickle.load(model)
    
    action_class = rf_model.predict(df_Features[x_columns])[0]
        
    return action_class

def alert(action_class):
    print("\n---------------------------------")
    if(action_class == 'F'):
        tprint("Fall detected", font = "small")
    else:
        tprint("Fall not detected", font = "small")


df_Features = compute_file(path, sampling_freq, trial_file_name)

if type(df_Features) != int:
    action_class = detection(df_Features)
    alert(action_class)
else:
    print("\nPlease introduce a valid file. Refer to the instructions.")