#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:11:14 2020

@author: arnautienda
"""

#models_AUC = {"1.5s": [] ,
 #                   "2s": [],
  #                  "3s": []}

# %% Libraries import

from os import walk

import pickle
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns

import configparser

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
import time


# %% Init config file

# Read config file

config = configparser.ConfigParser()
config.read('config.ini')

# %% Fallup processed files import

# Directories where data is stored with init file
unified_directory = config['paths_root']['unified_directory']

# %% Load files

##### Select window size for the study ######

#window_size = "1.5"
#window_size = "2"
window_size = "3"

colnames = ["na",'File', 'Fall_ADL', 'Act_Type', 'var_X', 'mean_X',
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
my_data_file_name = unified_directory + "Unified_Fallup_" + window_size + "s.txt"

# Create a Dataframe from the file with the engineered features previously created
fallup_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',',names = colnames, header = None))

fallup_data.drop("na", axis=1, inplace=True)


# We load the data created by us
my_data_file_name = unified_directory + "Unified_Pinetime_" + window_size + "s.txt"

# Create a Dataframe from the file with the engineered features previously created
pinetime_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

pinetime_data.drop("na", axis=1, inplace=True)

# We load UMA dataset

# We load the data created by us
my_data_file_name = unified_directory + "Unified_UMA_" + window_size + "s.txt"

# Create a Dataframe from the file with the engineered features previously created
uma_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

uma_data.drop("na", axis=1, inplace=True)

# We load ITA dataset

# We load the data created by us
#my_data_file_name = unified_directory + "Unified_Ita.txt"

# Create a Dataframe from the file with the engineered features previously created
#ita_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

#ita_data.drop("kk", axis=1, inplace=True)

# We load UCI dataset

# We load the data created by us
my_data_file_name = unified_directory + "Unified_Uci_" + window_size + "s.txt"

# Create a Dataframe from the file with the engineered features previously created
uci_data = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ',', names = colnames, header = None))

uci_data.drop("na", axis=1, inplace=True)

# Concatenate fallup data and real data with uma
df_ADL_Falls = pd.concat([fallup_data, pinetime_data, uma_data, uci_data])



#%% EDA

# Shape of dataframe
print(df_ADL_Falls.shape)

# Separate falls dataframe from daily activites
df_only_ADLs = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "D"]
df_only_Falls = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "F"]
plt.figure(figsize=(7, 7))
ax = sns.countplot(x="Fall_ADL", data=df_ADL_Falls)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    




a = df_ADL_Falls.isnull()
plt.figure(figsize=(7,5))
sns.heatmap(a,yticklabels=False)


l = df_ADL_Falls.iloc[:,3:].columns.values
number_of_columns=6
number_of_rows = 9
plt.figure(figsize=(number_of_columns*4,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.set_style('whitegrid')
    sns.boxplot(y = df_ADL_Falls.iloc[:,3:][l[i]],color='green')
    plt.tight_layout()
#%% Train and test datasets


# Separate falls dataframe from daily activites
df_only_ADLs = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "D"]
df_only_Falls = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "F"]

print(df_only_ADLs.tail())
print(df_only_Falls.tail())

# Create train and tests dataset
X_train, X_test, y_train, y_test= train_test_split(df_ADL_Falls.drop('Fall_ADL', axis = 1), df_ADL_Falls['Fall_ADL'], test_size=0.30, random_state=15)

# Show number of files
print("\nTotal ADL: " + str(len(df_only_ADLs)))
print("Total Falls: " + str(len(df_only_Falls)))
print("GRAND Total: " + str(len(df_only_Falls)+len(df_only_ADLs)))
print("________________________")
print("\n70% train, 30% test")

print("\nTrain Falls: "+ str(len(y_train[y_train == 'F'])))
print("Train ADL: "+ str(len(y_train[y_train == 'D'])))
print("Train TOTAL: "+ str(len(y_train)))

print("\nTest Falls: "+ str(len(y_test[y_test == 'F'])))
print("Test ADL: "+ str(len(y_test[y_test == 'D'])))
print("Test TOTAL: "+ str(len(y_test)))

print("________________________")

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

X_train = X_train[x_columns]
X_test = X_test[x_columns] 




# Train normalization
x = X_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train_norm = pd.DataFrame(x_scaled)

# Test normalization
x = X_test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_test_norm = pd.DataFrame(x_scaled)




# %% Grid search KNN

# Create kneighbors classifier
clf = KNeighborsClassifier()
            
# Set possible parameters
param_grid = {"n_neighbors": range(1, 11), "weights": ["uniform", "distance"]}

# Execute search with crossvalidation = 4
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)

#  Fit classifier
grid_search.fit(X_train, y_train)

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
clf_knn.fit(X_train, y_train)

# Run and print predictions
predictions_knn = clf_knn.predict(X_test)


#%% Grid Search RF

clf=RandomForestClassifier()

param_grid = {"n_estimators": range(1,200,20), "max_depth": range(1,10)}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)

grid_search.fit(X_train, y_train)
means = grid_search.cv_results_["mean_test_score"]
stds = grid_search.cv_results_["std_test_score"]
params = grid_search.cv_results_['params']

best_n_estimators = grid_search.best_params_['n_estimators']
best_max_depth =  grid_search.best_params_['max_depth']

for mean, std, pms in zip(means, stds, params):
    print("Precisió mitjana: {:.2f} +/- {:.2f} amb paràmetres {}".format(mean*100, std*100, pms))

print("Els millors paràmetres han sigut: {}".format(grid_search.best_params_))

# %% Random forest

#Create a Gaussian Classifier
clf_rf=RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf_rf.fit(X_train,y_train)

with open('RFT.pkl', 'wb') as RFT:
    pickle.dump(clf_rf, RFT)

predictions_rf=clf_rf.predict(X_test)
print(predictions_rf)


#%% Grid Search SVC

from sklearn.svm import SVC
clf = SVC()

param_grid={"C": range(1,10,1), 'kernel': ['rbf','linear','poly','sigmoid'], 'gamma': ['scale','auto']}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4)

grid_search.fit(X_train, y_train)
means = grid_search.cv_results_["mean_test_score"]
stds = grid_search.cv_results_["std_test_score"]
params = grid_search.cv_results_['params']

best_C =  grid_search.best_params_['C']
best_kernel =  grid_search.best_params_['kernel']
best_gamma =  grid_search.best_params_['gamma']

for mean, std, pms in zip(means, stds, params):
    print("Precisió mitjana: {:.2f} +/- {:.2f} amb paràmetres {}".format(mean*100, std*100, pms))

print("Els millors paràmetres han sigut: {}".format(grid_search.best_params_))


#%% SVC

clf_svc = SVC(C = best_C, kernel = best_kernel, gamma = best_gamma, probability = True)

# Fit/train classifier
clf_svc.fit(X_train, y_train)

# Run and print predictions
predictions_svc = clf_svc.predict(X_test)
print(predictions_svc)

# %% NN
from keras.callbacks import TensorBoard
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim=len(x_columns), activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

y_test_disc = y_test.astype('category').cat.codes.values
y_train_disc = y_train.astype('category').cat.codes.values
x_val = X_test_norm
y_val = y_test_disc


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# simple early stopping
callbacks = [
             ModelCheckpoint(filepath='best_model.h5', save_weights_only=True, monitor='val_loss', save_best_only=True)]
history_callback = model.fit(X_train_norm, y_train_disc,  epochs=500, batch_size = 32, validation_data=(X_test_norm, y_test_disc), callbacks = callbacks)

 #%% Save NN

model.load_weights('best_model.h5')

#%%

predictions = model.predict(X_test_norm)
predictions_nn = []
for i in range(0,len(predictions)):
    if(np.argmax(predictions[i]) == 0):
        predictions_nn.append("D")
    else:
        predictions_nn.append("F")


#%% ROC Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# models_AUC = {"1.5s": [] ,
#                    "2s": [],
#                    "3s": []}
clf = [clf_knn, clf_rf, clf_svc, model]
models = ["KNN", "RF", "SVC", "NN"]
fig, ax = plt.subplots()
fig.suptitle('Curva ROC con ventana de ' + window_size + 's', fontsize=14)
ns_probs = [0 for _ in range(len(y_test_disc))]
ns_auc = roc_auc_score(y_test_disc, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test_disc, ns_probs)
ax = plt.plot(ns_fpr, ns_tpr, linestyle='--', label= 'AUC = 0.5')
for clf, mod in zip(clf, models):
    if mod == "NN":
        lr_probs = clf.predict_proba(X_test_norm)
    else:
        lr_probs = clf.predict_proba(X_test)
 
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    lr_auc = roc_auc_score(y_test_disc, lr_probs)
    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test_disc, lr_probs)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, linestyle='--', label='AUC (' + mod + ') = ' + str(round(lr_auc,2)))
    models_AUC[window_size + "s"].append(round(lr_auc,2))
    
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()

# show the plot
plt.show()

#%% Confusion matrix
fig, ax = plt.subplots(2,2,figsize = (10,10))
predictions = [predictions_knn, predictions_rf, predictions_svc, predictions_nn]
models = ["KNN", "RF", "SVC", "NN"]
#models_scores = {"1.5s": [] ,
#                    "2s": [],
#                    "3s": []}
i = 0
for pred, mod in zip(predictions,models):
    cm = confusion_matrix(y_test, pred, labels=["F", "D"])
    print("Confusion Matrix: " + mod)
    print("-----------------")
    print(cm)
    print("-----------------")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix (Normalized):")
    print("-----------------------------")
    print(cm_norm)
    print("-----------------------------")
    
    # calculations of measurements of performance
    
    n_TP = cm[0,0]
    n_FP = cm[1,0]
    n_TN = cm[1,1]
    n_FN = cm[0,1]
    
    # SENSITIVITY = TP / (TP + FN)
    Sensitivity = n_TP / (n_TP + n_FN)
    print("Sensitivity = "+ str(Sensitivity))
    
    # SPECIFICITY = TN / (FP + TN)
    Specificity = n_TN / (n_FP + n_TN)
    print("Specificity = "+ str(Specificity))
    
    # Precision = TP / (TP + FP)
    Precision = n_TP / (n_TP + n_FP)
    print("Precision = "+ str(Precision))
    
    # Accuracy = (TP + TN) / (TP + FP + TN + FN)
    Accuracy = (n_TP + n_TN) / (n_TP + n_FP + n_TN + n_FN)
    print("Accuracy = "+ str(Accuracy))
    
    model_score = round(0.7 * Sensitivity + 0.3 * Specificity,2)
    print("MODEL SCORE: " + str(model_score))
    print("--------------------------------------")
    
    #fig.suptitle('Confusion matrix with ' + window_size + 's window size', fontsize=14)
    fig.suptitle('Matriz de confusión con ventana de ' + window_size + 's', fontsize=14)

    models_scores[window_size + "s"].append(model_score)
    ax.flatten()[i].set_title(mod)
    g = sns.heatmap(cm_norm, annot=True, ax = ax.flatten()[i],
                xticklabels = ["F","D"],
                yticklabels = ["F","D"])
    g = g.set(xlabel='Clase verdadera', ylabel='Clase predicha')
    i+=1
#%% Results (Execute once all windows sizes are examinated)

results = pd.DataFrame({"Model": models, 
                        "1.5s": models_scores["1.5s"],
                        "2s": models_scores["2s"],
                        "3s": models_scores["3s"]})
                       
#%% Results AUC (Execute once all windows sizes are examinated)

results_AUC = pd.DataFrame({"Model": models, 
                        "1.5s": models_AUC["1.5s"],
                        "2s": models_AUC["2s"],
                        "3s": models_AUC["3s"]})
                       
#%% Labels legend
labels = ['TP','FP','FN','TN']
labels = np.asarray(labels).reshape(2,2)
g = sns.heatmap(cm, annot=labels, fmt='',xticklabels = ["F","D"],yticklabels = ["F","D"])
g = g.set(xlabel='Clase verdadera', ylabel='Clase predicha')
