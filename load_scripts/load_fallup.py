#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:09:14 2020

@author: arnautienda
"""
# %% Libraries import

import pandas as pd
import numpy as np
import configparser
import math
import os
from os import walk


# %% Init config file

# Read config file

config = configparser.ConfigParser()
config.read('config.ini')

# %% Fallup processed files import

# Directories where data is stored with init file
unified_directory = config['paths_root']['unified_directory']
generated_datasets_directory = config['paths_fallup']['generated_datasets_directory']

file_names = []
dir_names = []

# Scanning of the directory and subdirectories
for (dirpath, dirnames, filenames) in walk(generated_datasets_directory):
    file_names.extend(filenames)

# Dataframe from the file names and create a column with these names
df_Files_Trials = pd.DataFrame({"File": file_names})

# %% Engineering features

# The following function generates over 50 engineered features from the X,Y,Z reading.
# We save de results in a txt, in which every line is a processed file, and the columns
# are the 50 features we have created.

# INPUT: File name

# OUTPUT: A line is written in the unified file


def compute_trial_file(trial_file_name, window_size):
    
    # Create a dataframe from the csv file  
    df_Mediciones_file = pd.DataFrame(pd.read_csv(generated_datasets_directory + trial_file_name, header = 0, sep = ',', 
                                         names = ["","Subject","Act_Type","Trial",
                                                  "X","Y","Z"], 
                                                   skiprows= 0))
    # We just need the triaxial information
    df_Mediciones_aux = df_Mediciones_file[["X","Y","Z"]]
    
    sf = 20 # sampling frequency of fallup study
    ws = window_size # window size in seconds
    ws_samples = sf * ws # window size in samples
    df_Mediciones = df_Mediciones_aux

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
    

    # We work with the window size defined above
    
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
    
    # Extract the activity Type from file's name
    act = ""
    if(df_Mediciones_file.iloc[0,2] < 6):
        act = "F"
    else:
        act = "D"
    df_Features_2 = pd.DataFrame({"File": [trial_file_name],
                                  "Fall_ADL": act,
                                  "Act_Type": [trial_file_name[4:6]],
                                 })  

    df_Features =  pd.concat([df_Features_2, df_Features], axis=1)

    # writes the record/instance data:
    df_Features.to_csv(unified_directory + 'Unified_Fallup_' + str(ws) + 's.txt', mode='a', header=False)
    
   # del df_Features
    del df_Features_2
    
    return max_N, max_N_index, min_N, min_N_index, df_Mediciones_file


"""______________________________________________________________ 
"""
# import libraries to monitor time-computing progress...

import time
from datetime import timedelta

start_time = time.time()


file_list = df_Files_Trials[["File"]]
total_num_iter = len(file_list)




# We iterate over the file_list we extracted by scanning the directories and subdirectories  
for window_size in [1.5,2,3]:  
    iter_no = 1
    try:
        # Delete the file before creating a new one
        os.remove(unified_directory + 'Unified_Fallup_' + str(window_size) + 's.txt')
    except:
        print("No files to remove")
        
    for index, row in file_list.iterrows():
        iter_start_time = time.time()
        
        my_data_file_name = row['File']
        print("________________________________________")  
        print("ITERATION NO: " + str(iter_no) + "/" + str(total_num_iter))
        iter_no +=1
    
        # Exclude hidden files
        if(row['File'] == '.DS_Store'):
            continue
            
        print("PROCESSING TRIAL FILE: " + row['File'])
        max_N, max_N_index, min_N, min_N_index, df = compute_trial_file(my_data_file_name, window_size)
      
        elapsed_time_secs = time.time() - iter_start_time
        msg = "Iteration took: %s secs" % timedelta(seconds=round(elapsed_time_secs))
        print(msg)
        remaining_time = round(((time.time() - start_time)/iter_no)*(total_num_iter-iter_no))
        msg = "REMAINING TIME: %s secs" % timedelta(seconds = remaining_time)
        print(msg)
    
        print("________________________________________")    
        
    
    elapsed_time_secs = time.time() - start_time
    
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    
    print(msg)