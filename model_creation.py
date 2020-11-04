#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:11:14 2020

@author: arnautienda
"""

# %% Libraries import

from os import walk

import pickle
import pandas as pd
import numpy as np
import scipy as sc

import configparser

import matplotlib.pyplot as plt


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# %% Init config file

# Read config file

config = configparser.ConfigParser()
config.read('config.ini')

# %% Fallup processed files import

# Directories where data is stored with init file
fallup_directory = config['paths_fallup']['fallup_directory']
eng_feat_directory = config['paths_root']['eng_feat_directory']
generated_datasets_directory = config['paths_root']['generated_datasets_directory']

file_names = []
dir_names = []

# Recursively scan the directory and sub-directory for filenames...
for (dirpath, dirnames, filenames) in walk(generated_datasets_directory):
    file_names.extend(filenames)



# Create a dataframe and load filenames into the File column
df_Files_Trials = pd.DataFrame({"File": file_names})

# %%

colnames = ["kk",'File', 'Fall_ADL', 'Act_Type', 'var_X', 'mean_X',
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

# We work with the prepared file Unified_ADL_Falls, which is based on the previous dataset
my_data_file_name = eng_feat_directory + "Unified_Fallup.txt"

# Create a Dataframe from the file with the engineered features previously created
fallup_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',',names = colnames, header = None))

fallup_data.drop("kk", axis=1, inplace=True)


# We load the data created by us
my_data_file_name = eng_feat_directory + "Unified_Real_Data.txt"


# Create a Dataframe from the file with the engineered features previously created
real_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

real_data.drop("kk", axis=1, inplace=True)

# We load UMA dataset

# We load the data created by us
my_data_file_name = eng_feat_directory + "Unified_UMA.txt"

# Create a Dataframe from the file with the engineered features previously created
uma_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

uma_data.drop("kk", axis=1, inplace=True)

# We load ITA dataset

# We load the data created by us
my_data_file_name = eng_feat_directory + "Unified_Ita.txt"

# Create a Dataframe from the file with the engineered features previously created
ita_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

ita_data.drop("kk", axis=1, inplace=True)

# We load UCI dataset

# We load the data created by us
my_data_file_name = eng_feat_directory + "Unified_Uci.txt"

# Create a Dataframe from the file with the engineered features previously created
uci_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

uci_data.drop("kk", axis=1, inplace=True)

# Concatenate fallup data and real data with uma
df_ADL_Falls = pd.concat([fallup_data,real_data,uma_data,uci_data])

# Separate falls dataframe from daily activites
df_only_ADLs = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "D"]
df_only_Falls = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "F"]

print(df_only_ADLs.tail())
print(df_only_Falls.tail())

#%% Train and test datasets

# Create train and tests dataset
X_train, X_test, y_train, y_test= train_test_split(df_ADL_Falls.drop('Fall_ADL', axis = 1), df_ADL_Falls['Fall_ADL'], test_size=0.30, random_state=15)

# Show number of files
print("Total ADL: " + str(len(df_only_ADLs)))
print("Total Falls: " + str(len(df_only_Falls)))
print("GRAND Total: " + str(len(df_only_Falls)+len(df_only_ADLs)))
print("---------------------------------------")
print("Train Falls: "+ str(len(y_train[y_train == 'F'])))
print("Train ADL: "+ str(len(y_train[y_train == 'D'])))
print("Train TOTAL: "+ str(len(y_train)))
print("---------------------------------------")
print("Test Falls: "+ str(len(y_test[y_test == 'F'])))
print("Test ADL: "+ str(len(y_test[y_test == 'D'])))
print("Test TOTAL: "+ str(len(y_test)))

#%% Train and test separation

# Original features
x_columns = ['kurtosis_X','max_X','mean_X','min_X','range_X','skewness_X','std_X','var_X',
             'kurtosis_Y','max_Y','mean_Y','min_Y','range_Y','skewness_Y','std_Y','var_Y',
             'kurtosis_Z','max_Z','mean_Z','min_Z','range_Z','skewness_Z','std_Z','var_Z',
             'kurtosis_N_XYZ','max_N_XYZ','mean_N_XYZ','min_N_XYZ','range_N_XYZ','skewness_N_XYZ','std_N_XYZ','var_N_XYZ',
             'kurtosis_N_HOR','max_N_HOR','mean_N_HOR','min_N_HOR','range_N_HOR','skewness_N_HOR','std_N_HOR','var_N_HOR',
             'kurtosis_N_VER','max_N_VER','mean_N_VER','min_N_VER','range_N_VER','skewness_N_VER','std_N_VER','var_N_VER',
             'corr_HV','corr_NH','corr_NV','corr_XY','corr_XZ','corr_YZ']

# Features simplified
x_columns_simple = ['max_X','mean_X','min_X','range_X','std_X','var_X',
					'max_Y','mean_Y','min_Y','range_Y','std_Y','var_Y',
					'max_Z','mean_Z','min_Z','range_Z','std_Z','var_Z',
					'max_N_XYZ','mean_N_XYZ','min_N_XYZ','range_N_XYZ','std_N_XYZ','var_N_XYZ'
					]


# prepare/get the columns
X_train_simple = X_train[x_columns_simple]
X_test_simple = X_test[x_columns_simple] 


""" 
# Train normalization
x = X_train_simple.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train_simple_norm = pd.DataFrame(x_scaled)

# Test normalization
x = X_test_simple.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_test_simple_norm = pd.DataFrame(x_scaled)
"""



# %% Grid search KNN

# Create kneighbors classifier
clf = KNeighborsClassifier()
            
# Set possible parameters
param_grid = {"n_neighbors": range(1, 11), "weights": ["uniform", "distance"]}

# Execute search with crossvalidation = 4
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)

#  Fit classifier
grid_search.fit(X_train_simple, y_train)

# Save results
means = grid_search.cv_results_["mean_test_score"]
stds = grid_search.cv_results_["std_test_score"]
params = grid_search.cv_results_['params']
best_n_neighbors = grid_search.best_params_['n_neighbors']
best_weights =  grid_search.best_params_['weights']

# Print results
for mean, std, pms in zip(means, stds, params):
    print("Precisió mitjana: {:.2f} +/- {:.2f} amb paràmetres {}".format(mean*100, std*100, pms))

print("Els millors paràmetres han sigut: {}".format(grid_search.best_params_))



# %% Knn with best params

# knn classification with best parameters of grid search
clf_knn = KNeighborsClassifier(n_neighbors = best_n_neighbors, weights = best_weights)

# Fit/train classifier
clf_knn.fit(X_train_simple, y_train)

# Run and print predictions
predictions = clf_knn.predict(X_test_simple)
print(predictions)



#%% Grid Search RF

clf=RandomForestClassifier()

param_grid = {"n_estimators": range(1,200,20), "max_depth": range(1,10)}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)

grid_search.fit(X_train_simple, y_train)
means = grid_search.cv_results_["mean_test_score"]
stds = grid_search.cv_results_["std_test_score"]
params = grid_search.cv_results_['params']

for mean, std, pms in zip(means, stds, params):
    print("Precisió mitjana: {:.2f} +/- {:.2f} amb paràmetres {}".format(mean*100, std*100, pms))

print("Els millors paràmetres han sigut: {}".format(grid_search.best_params_))

#%% Grid Search SVC
from sklearn.svm import SVC
clf = SVC()

param_grid={"C": range(1,10,1), 'kernel': ['rbf','linear','poly','sigmoid'], 'gamma': ['scale','auto']}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)

grid_search.fit(X_train_simple, y_train)
means = grid_search.cv_results_["mean_test_score"]
stds = grid_search.cv_results_["std_test_score"]
params = grid_search.cv_results_['params']

for mean, std, pms in zip(means, stds, params):
    print("Precisió mitjana: {:.2f} +/- {:.2f} amb paràmetres {}".format(mean*100, std*100, pms))

print("Els millors paràmetres han sigut: {}".format(grid_search.best_params_))


#%% SVC

clf_svc = SVC(C = 9, kernel = 'rbf', gamma = 'auto')

# Fit/train classifier
clf_svc.fit(X_train_simple, y_train)

# Run and print predictions
predictions_svc = clf_svc.predict(X_test_simple)
print(predictions_svc)
#%% Knn

clf_knn = KNeighborsClassifier(n_neighbors = 6, weights = 'distance')
# Fit/train classifier
clf_knn.fit(X_train_simple, y_train)

# Run and print predictions
predictions = clf_knn.predict(X_test_simple)
print(predictions)

#%% SVC

# Here we use LinearSVC
from sklearn.svm import LinearSVC

# dcefine the classifier
clf_sv = LinearSVC(random_state=0, tol=1e-8, dual=False)

# Fit/train classifier
clf_svc.fit(X_train_simple_norm, y_train)

# Run and print predictions
predictions = clf_svc.predict(X_test_simple_norm)
print(predictions)

with open('SVC.pkl', 'wb') as SVC:
    pickle.dump(clf, SVC)


# %% Random forest

#Create a Gaussian Classifier
clf_rf=RandomForestClassifier(n_estimators=80,max_depth=9)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf_rf.fit(X_train_simple,y_train)

with open('RFT.pkl', 'wb') as RFT:
    pickle.dump(clf_rf, RFT)

predictions_rf=clf_rf.predict(X_test_simple)
print(predictions_rf)


# %% NN

import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


dataset = tf.data.Dataset.from_tensor_slices((X_train_simple_norm.values, y_train.astype('category').cat.codes.values))

train_dataset = dataset.shuffle(len(X_train_simple_norm)).batch(1)


#print(train_dataset)
model.fit(train_dataset, epochs=350)

 #%% Save NN

with open('NN_model.pkl', 'wb') as NN_model:
    pickle.dump(model, NN_model)
#%%


predictions = my_model_NN.predict(X_test_simple_norm)
predictions_class = []
for i in range(0,len(predictions)):
    print(np.argmax(predictions[i]))
    if(np.argmax(predictions[i]) == 0):
        predictions_class.append("D")
    else:
        predictions_class.append("F")


#%% 
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf_rf, X_test_simple, y_test, normalize = 'true')
#%%

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
cm = confusion_matrix(y_test, predictions_svc, labels=["D", "F"])
print("Confusion Matrix:")
print("-----------------")
print(cm)
print("-----------------")
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix (Normalized):")
print("-----------------------------")
print(cm_norm)
print("-----------------------------")
y_test.astype('category').cat.codes.values

y_test_binary = y_test.astype('category').cat.codes.values
predictions_binary = pd.Series(predictions_svc).astype('category').cat.codes.values

fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, predictions_binary)
print(metrics.auc(fpr, tpr))
fig, ax = plt.subplots()
viz = plot_roc_curve(clf_rf, X_test_simple, y_test,
                     name = 'ROC_RF',
                     alpha=0.3, lw=1, ax=ax)

viz = plot_roc_curve(clf_knn, X_test_simple, y_test,
                     name = 'ROC_KNN',
                     alpha=0.3, lw=1, ax=ax)

viz = plot_roc_curve(clf_svc, X_test_simple, y_test,
                     name = 'ROC_SVC',
                     alpha=0.3, lw=1, ax=ax)
#%%

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

    # %% Check errors

X_test['predicted_class'] = predictions
X_test['class'] = y_test
