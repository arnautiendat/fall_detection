#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:09:14 2020

@author: arnautienda
"""
# %% Libraries import

import pickle
import pandas as pd
import numpy as np
import scipy as sc
import configparser
from os import walk



# %% Init config file

# Read config file

config = configparser.ConfigParser()
config.read('config.ini')

# %% Uma processed files import

# Preliminary step 0. We need to establish/select our working folders. 
ita_directory = config['paths_ita']['ita_directory']
eng_feat_directory = config['paths_root']['eng_feat_directory']

file_names = []
dir_names = []

# Recursively scan the directory and sub-directory for filenames...
for (dirpath, dirnames, filenames) in walk(ita_directory):
    file_names.extend(filenames)
    for filename in filenames:
        dir_names.append(dirpath + '/' + filename)
    

# Create a dataframe and load filenames into the File column
df_Files_Trials = pd.DataFrame({"File": file_names, "Dirs": dir_names})

#%% Carreguem un fitxer

dir_name = dir_names[10]
df_Mediciones_all = pd.read_csv(dir_name, header = 0, sep = ';', 
                                     names = ['Time(s)','X','Y','Z',"kk","kk1","kk2","kk3","kk4","kk5","kk6","kk7","kk8","kk9"],skiprows = 1)




# %% Engineering features


# The following function processes a trial file to compute and add 50+ metrics (engineered features)
# the results are saved in a file, which will be used to train several ML models.
# We work with a window of 600 measurements (equivalent to 3 seconds of activity recording)
# We handle three scenarios, the window falls completely within the values, or the windows is aligned
# with the minimum/maximum value...
# OUTPUT: We write the 
# Note: this function have a similar structure as the read_trial() function above.
def compute_trial_file(trial_file_name, dir_name):
 
    
    df_Mediciones_all = pd.read_csv(dir_name, header = 0, sep = ';', 
                                     names = ['Time(s)','X','Y','Z',"kk","kk1","kk2","kk3","kk4","kk5","kk6","kk7","kk8","kk9"],skiprows = 1)

    
    df_Mediciones_aux = df_Mediciones_all[["X","Y","Z"]]
    df_Mediciones_aux = df_Mediciones_aux.rename(columns = {"X": "S1_X","Y": "S1_Y","Z": "S1_Z"})
    """
    ---------------------------------------------------
    Note: extract from SisFall Dataset info, 
          about the accelerometer sensors:
    ---------------------------------------------------
    Data are in bits with the following characteristics:
    
    In order to convert the acceleration data (AD) given 
    in bits into gravity, use this equation: 
        Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
    
    In order to convert the rotation data (RD) given in 
    bits into angular velocity, use this equation:
        Angular velocity [°/s]: [(2*Range)/(2^Resolution)]*RD
    ---------------------------------------------------
    """
    sf = 33 # sampling frequency
    ws = 3 # window size in seconds
    ws_samples = sf * ws # window size in samples
    df_Mediciones = df_Mediciones_aux
    #df_Mediciones = df_Mediciones_aux.iloc[::ds_factor, :]
    
    # Calculate the values for SENSOR_1
    import math
    Sensor1_Resolution = 13
    Sensor1_Range = 16
    g = (2*Sensor1_Range/2**Sensor1_Resolution)

    # Calculate the Euclidean Norm of (XYZ) Acceleration
    fn = lambda row: math.sqrt((row.S1_X)**2 + (row.S1_Y)**2 + (row.S1_Z)**2) 
    col = df_Mediciones.apply(fn, axis=1) 
    df_Mediciones = df_Mediciones.assign(S1_N_XYZ=col.values)   

    # Get the max value of N_XYZ
    # This max value is used as a reference point to get a window of values
    S1_max_N_XYZ = np.max(df_Mediciones.S1_N_XYZ)
    print("S1_max_N_XYZ = " + str(S1_max_N_XYZ))

    max_N = np.max(df_Mediciones.S1_N_XYZ)
    max_N_index = df_Mediciones.index[df_Mediciones.S1_N_XYZ == max_N][0]
    min_N = np.min(df_Mediciones.S1_N_XYZ)
    min_N_index = df_Mediciones.index[df_Mediciones.S1_N_XYZ == min_N][0]
    len_df_Mediciones = len(df_Mediciones)
    

    # We work with a window of 600 measurements (equivalent to 3 seconds of activity recording)
    # We handle three scenarios, the window falls completely within the values, or the windows is aligned
    # with the minimum/maximum value...
    if (max_N_index - round(ws_samples/2)<0):
        df_Mediciones = df_Mediciones[0:round(ws_samples+1)]
        print("LOW WINDOW")
    else:
        if (max_N_index + round(ws_samples/2)+1> len_df_Mediciones):
            df_Mediciones = df_Mediciones[len_df_Mediciones-round(ws_samples+2):len_df_Mediciones-1]
            print("HIGH WINDOW")
        else:
            # extract the central window
            df_Mediciones = df_Mediciones[max_N_index - round(ws_samples/2): max_N_index + round(ws_samples/2)+1]
            print("NORMAL WINDOW")
            
    print("max_N = " + str(max_N))
    print("max_N_index = " + str(max_N_index))
    print("min_N = " + str(min_N))
    print("min_N_index = " + str(min_N_index))    
           
    # S1_N_HOR: Calculate the Euclidean Norm of (HOR) Acceleration
    fn_hor = lambda row: math.sqrt((row.S1_Y)**2 + (row.S1_Z)**2) # define a function for the new column
    col = df_Mediciones.apply(fn_hor, axis=1) # get column data with an index
    df_Mediciones = df_Mediciones.assign(S1_N_HOR=col.values) # assign values to column 'c'  

    # S1_N_VER: Calculate the Euclidean Norm of (VER) Acceleration
    fn_ver = lambda row: math.sqrt((row.S1_X)**2 + (row.S1_Z)**2) # define a function for the new column
    col = df_Mediciones.apply(fn_ver, axis=1) # get column data with an index
    df_Mediciones = df_Mediciones.assign(S1_N_VER=col.values) # assign values to column 'c' 
    
    field_name = "S1_X"

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

    field_name = "S1_Y"
    
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

    field_name = "S1_Z"

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
    field_name = "S1_N_XYZ"

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

    field_name = "S1_N_HOR"

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

    field_name = "S1_N_VER"

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

    corr_valueXY = df_Mediciones["S1_X"].corr(df_Mediciones["S1_Y"])
    corr_valueXZ = df_Mediciones["S1_X"].corr(df_Mediciones["S1_Z"])
    corr_valueYZ = df_Mediciones["S1_Y"].corr(df_Mediciones["S1_Z"])
    corr_valueNV = df_Mediciones["S1_N_XYZ"].corr(df_Mediciones["S1_N_VER"])
    corr_valueNH = df_Mediciones["S1_N_XYZ"].corr(df_Mediciones["S1_N_HOR"])
    corr_valueHV = df_Mediciones["S1_N_HOR"].corr(df_Mediciones["S1_N_VER"])

    df_Features_2 = pd.DataFrame({"corr_XY": [corr_valueXY], 
                                "corr_XZ": [corr_valueXZ],
                                "corr_YZ": [corr_valueYZ],
                                "corr_NV": [corr_valueNV],
                                "corr_NH": [corr_valueNH], 
                                "corr_HV": [corr_valueHV]})       

    df_Features =  pd.concat([df_Features, df_Features_2], axis=1)

    if(iter_no < 127):
        act = "D"
    else:
        act = "F"
    df_Features_2 = pd.DataFrame({"File": [trial_file_name],
                                  "Fall_ADL": act,
                                  "Act_Type": act,
                                 })  


    df_Features =  pd.concat([df_Features_2, df_Features], axis=1)

    # writes the record/instance data:
    df_Features.to_csv(eng_feat_directory + 'Unified_Ita.txt', mode='a', header=False)
    
    del df_Features
    del df_Features_2
    
    return max_N, max_N_index, min_N, min_N_index


"""______________________________________________________________ 
"""
# import libraries to monitor time-computing progress...
import time
from datetime import timedelta

start_time = time.time()

#
# The following section, performs lots of computations.
# So, a few "traces" (print's) are included to monitor progress...
#

# NOTE: to filter a specific type of activity use the following line with the corresponding activity code
# for example to filter (and only process) ADL type D01:
# file_list = df_Files_Trials[df_Files_Trials.Act_Type == "D01"][["File"]]
# Otherwise the follwing line process all types of ADL and Falls:
file_list = df_Files_Trials
total_num_iter = len(file_list)
iter_no = 1
    
for index, row in file_list.iterrows():
    iter_start_time = time.time()
    
    my_data_file_name = row['File']
    my_data_path = row['Dirs']
    print("_________ ITERATION NO: " + str(iter_no) + "/" + str(total_num_iter) + " (TOTAL)___________________________")
    iter_no +=1
#   if iter_no< 499: 
#        print("_________ SKIPPING TRIAL FILE: " + row['File'] + " ___________________________")
#        continue
    if(row['File'] == '.DS_Store'):
        continue
    print("_________ PROCESSING TRIAL FILE: " + row['File'] + " ___________________________")
    max_N, max_N_index, min_N, min_N_index = compute_trial_file(my_data_file_name,my_data_path)
    print("________________________________________________________________________________________")    

    elapsed_time_secs = time.time() - iter_start_time
    msg = "Iteration took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)
    remaining_time = round(((time.time() - start_time)/iter_no)*(total_num_iter-iter_no))
    msg = "REMAINING TIME: %s secs (Wall clock time)" % timedelta(seconds = remaining_time)
    print(msg)

    print("________________________________________________________________________________________")    
    

elapsed_time_secs = time.time() - start_time

msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

print(msg)