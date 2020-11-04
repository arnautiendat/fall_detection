# %% Libraries
from sklearn.svm import LinearSVC
import pandas as pd
import pickle
from os import walk
import configparser



# %% Init config file

# Read config file

config = configparser.ConfigParser()
config.read('config_rasp.ini')



# %% Define paths and load file names
proves_directory = config['paths']['proves_directory']
eng_feat_directory = config['paths']['eng_feat_directory']


# The dataset files must be located in the "SisFall_dataset" folder 
file_names = []
dir_names = []

# Recursively scan the directory and sub-directory for filenames...
for (dirpath, dirnames, filenames) in walk(proves_directory):
    file_names.extend(filenames)

# Create a dataframe and load filenames into the File column
df_Files_Trials = pd.DataFrame({"File": file_names})

# %% Load file and engineering features

# The following function processes a trial file to compute and add 50+ metrics (engineered features)
# the results are saved in a file, which will be used to train several ML models.
# We work with a window of 600 measurements (equivalent to 3 seconds of activity recording)
# We handle three scenarios, the window falls completely within the values, or the windows is aligned
# with the minimum/maximum value...
# OUTPUT: We write the 
# Note: this function have a similar structure as the read_trial() function above.
def compute_trial_file(trial_file_name):
 
   # df_Mediciones = pd.DataFrame(pd.read_csv(trial_file_name, header = None, sep = ',', 
  #                                       names = ["X", "Y", "Z",
   #                                               "S2_X", "S2_Y", "S2_Z", 
    #                                              "S3_X", "S3_Y", "S3_Z"],skiprows= 0))
    
    
    df_Mediciones = pd.DataFrame(pd.read_csv(proves_directory + trial_file_name, header = None, sep = ',', 
											 names = ["X_nog", "Y_nog", "Z_nog","kk"], 
													 skiprows= 0))
    
    df_Mediciones = df_Mediciones.drop(["kk"],axis = 1)
    print(df_Mediciones)
    

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
    sf = 20 # sampling frequency
    ws = 3 # window size in seconds
    ws_samples = sf * ws # window size in samples
    # Calculate the values
    import math
    Resolution = 16
    Range = 4
    g = (2*Range/2**Resolution)


    df_Mediciones['X']= df_Mediciones['X_nog']*g
    df_Mediciones['Y']= df_Mediciones['Y_nog']*g
    df_Mediciones['Z']= df_Mediciones['Z_nog']*g

    # Calculate the Euclidean Norm of (XYZ) Acceleration
    fn = lambda row: math.sqrt((row.X)**2 + (row.Y)**2 + (row.Z)**2) 
    col = df_Mediciones.apply(fn, axis=1) 
    df_Mediciones = df_Mediciones.assign(N_XYZ=col.values)   


    # Get the max value of N_XYZ
    # This max value is used as a reference point to get a window of values
    max_N_XYZ = np.max(df_Mediciones.N_XYZ)
    print("max_N_XYZ = " + str(max_N_XYZ))

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
    
           
    # N_HOR: Calculate the Euclidean Norm of (HOR) Acceleration
    fn_hor = lambda row: math.sqrt((row.Y)**2 + (row.Z)**2) # define a function for the new column
    col = df_Mediciones.apply(fn_hor, axis=1) # get column data with an index
    df_Mediciones = df_Mediciones.assign(N_HOR=col.values) # assign values to column 'c'  

    # N_VER: Calculate the Euclidean Norm of (VER) Acceleration
    fn_ver = lambda row: math.sqrt((row.X)**2 + (row.Z)**2) # define a function for the new column
    col = df_Mediciones.apply(fn_ver, axis=1) # get column data with an index
    df_Mediciones = df_Mediciones.assign(N_VER=col.values) # assign values to column 'c' 
    
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

    act = ""
    if(int(trial_file_name[4:6]) < 6):
        act = "F"
    else:
        act = "D"
    df_Features_2 = pd.DataFrame({"File": [trial_file_name],
                                  "Fall_ADL": act,
                                  "Act_Type": [trial_file_name[4:6]],
                                 })  

    df_Features =  pd.concat([df_Features_2, df_Features], axis=1)

    # writes the record/instance data:
    df_Features.to_csv('Unified_Real_Data.txt', mode='a', header=False)
    
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
file_list = df_Files_Trials[["File"]]
total_num_iter = len(file_list)
iter_no = 0
    
for index, row in file_list.iterrows():
    iter_start_time = time.time()
    
    my_data_file_name = row['File']
    print("_________ ITERATION NO: " + str(iter_no) + "/" + str(total_num_iter) + " (TOTAL)___________________________")
    iter_no +=1
#    if iter_no< 499: 
#        print("_________ SKIPPING TRIAL FILE: " + row['File'] + " ___________________________")
#        continue
    if(row['File'] == '.DS_Store'):
        continue    
    print("_________ PROCESSING TRIAL FILE: " + row['File'] + " ___________________________")
    max_N, max_N_index, min_N, min_N_index = compute_trial_file(my_data_file_name)
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



# %% Predict


# We work with the prepared file Unified_ADL_Falls, which is based on the previous dataset
my_data_file_name = eng_feat_directory + "Unified_Real_Data.txtt"

colnames = ['File', 'Fall_ADL', 'Act_Type', 'var_X', 'mean_X',
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

# Create a Dataframe from the file with the engineered features previously created
data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))


#data.drop('0', axis=1, inplace=True)
"""
# Add column names
data.columns = ['File', 'Fall_ADL', 'Act_Type', 'var_X', 'mean_X',
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
"""

# Columns selected for the classifier
x_columns_simple = ['max_X','mean_X','min_X','range_X','std_X','var_X',
					'max_Y','mean_Y','min_Y','range_Y','std_Y','var_Y',
					'max_Z','mean_Z','min_Z','range_Z','std_Z','var_Z',
					'max_N_XYZ','mean_N_XYZ','min_N_XYZ','range_N_XYZ','std_N_XYZ','var_N_XYZ'
					]

data_simple = data[x_columns_simple]

clf = pickle.load(open("RFT.pkl", "rb" ))
# Run and print predictions
predictions_SVC = clf.predict(data_simple)
print(predictions_SVC)

#%%

data['predicted_class'] = predictions_SVC

data = data[['File', 'Fall_ADL','predicted_class', 'Act_Type', 'var_X', 'mean_X',
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
       'corr_NH', 'corr_HV']]
# %% Confusion Matrix
y_column = "Fall_ADL"

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data['Fall_ADL'], predictions_SVC, labels=["D", "F"])
print("Confusion Matrix:")
print("-----------------")
print(cm)
print("-----------------")
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix (Normalized):")
print("-----------------------------")	
print(cm_norm)
print("-----------------------------")


# %% Performance indicators

# calculations of measurements of performance

n_TP = cm[1,1]
n_FP = cm[1,0]
n_TN = cm[0,0]
n_FN = cm[0,1]

# SENSITIVITY = TP / (TP + FN)
svc_Sensitivity = n_TP / (n_TP + n_FN)
print("svc_Sensitivity = "+ str(svc_Sensitivity))

# SPECIFICITY = TN / (FP + TN)
svc_Specificity = n_TN / (n_FP + n_TN)
print("svc_Specificity = "+ str(svc_Specificity))

# Precision = TP / (TP + FP)
svc_Precision = n_TP / (n_TP + n_FP)
print("svc_Precision = "+ str(svc_Precision))

# Accuracy = (TP + TN) / (TP + FP + TN + FN)
svc_Accuracy = (n_TP + n_TN) / (n_TP + n_FP + n_TN + n_FN)
print("svc_Accuracy = "+ str(svc_Accuracy))

