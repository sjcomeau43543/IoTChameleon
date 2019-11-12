# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 00:01:10 2018

@author: deanecke
"""
import pandas as pd
import numpy as np
import os
from sklearn import tree, ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class ExperimentsManager:
    def __init__(self, dataset_csv):
        use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))
        cols_to_drop = ['device_category']
        y_col = 'device_category'
        test = pd.read_csv(os.path.abspath(dataset_csv), usecols=use_cols, low_memory=False)
        test = shuffle(test)
        self.test = self.clear_missing_data(test)
        self.x_test = self.perform_feature_scaling(self.test.drop(cols_to_drop, 1))
        self.y_test = self.test[y_col]
        self.devices = self.y_test.unique()

    def experiment_random_forest(self):
        """
        Running all the different random forest classifiers for all devices and prints
        their AUC value accordingly 
        """
        for device in self.devices:
            for criterion_name in ['gini','entropy']:
                for forest_size in [3,7,11,15,19,21,23]:
                    clf = self.load_model_from_pkl(r"C:\Users\deanecke\Documents\Project_updated\IoT-device-type-identification-master\models\{0}\{0}_forest_{1}_{2}.pkl".format(device,criterion_name,forest_size))
                    y_test = np.array(pd.Series(self.get_is_dev_vec(device, self.y_test)))
                    pred = np.array(clf.predict(self.x_test))
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
                    print("DEVICE:{0},CRITERION:{1},SIZE:{2}".format(device, criterion_name, forest_size))
                    print(metrics.auc(fpr, tpr))
                    
    def experiment_decision_trees(self):
        """
        Running all the different decision trees classifiers for all devices and prints
        their AUC value accordingly 
        """
        for device in self.devices:
            for criterion_name in ['gini','entropy']:
                for samples_size in [50,100,200,400]:
                    clf = self.load_model_from_pkl(r"C:\Users\deanecke\Documents\Project_updated\IoT-device-type-identification-master\models\{0}\{0}_cart_{1}_{2}_samples_leaf.pkl".format(device,criterion_name,samples_size))
                    y_test = np.array(pd.Series(self.get_is_dev_vec(device, self.y_test)))
                    pred = np.array(clf.predict(self.x_test))
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
                    print("DEVICE:{0},CRITERION:{1},SIZE:{2}".format(device, criterion_name, samples_size))
                    print(metrics.auc(fpr, tpr))
                    
    def experiment_knn(self):
        """
        Running the KNN (K Nearest Neighbours) classifier for 
        each device 
        """
        for device in self.devices:
            clf = self.load_model_from_pkl(r"C:\Users\deanecke\Documents\Project_updated\IoT-device-type-identification-master\models\{0}\{0}_knn_5_uniform.pkl".format(device))
            y_test = np.array(pd.Series(self.get_is_dev_vec(device, self.y_test)))
            pred = np.array(clf.predict(self.x_test))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
            print("DEVICE:{0}".format(device))
            print(metrics.auc(fpr, tpr))
            
    def experiment_sgd(self):
        """
        Running the SGD (Stochastic Gradient Descent) classifier for 
        each device 
        """
        for device in self.devices:
            clf = self.load_model_from_pkl(r"C:\Users\deanecke\Documents\Project_updated\IoT-device-type-identification-master\models\{0}\{0}_sgd.pkl".format(device))
            y_test = np.array(pd.Series(self.get_is_dev_vec(device, self.y_test)))
            pred = np.array(clf.predict(self.x_test))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
            print("DEVICE:{0}".format(device))
            print(metrics.auc(fpr, tpr))
            
    def experiment_naive_bayes(self):
        """
        Running the Gaussian Naive Bayes classifier for 
        each device 
        """
        for device in self.devices:
            clf = self.load_model_from_pkl(r"C:\Users\deanecke\Documents\Project_updated\IoT-device-type-identification-master\models\{0}\{0}_naive_bayes.pkl".format(device))
            y_test = np.array(pd.Series(self.get_is_dev_vec(device, self.y_test)))
            pred = np.array(clf.predict(self.x_test))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
            print("DEVICE:{0}".format(device))
            print(metrics.auc(fpr, tpr))
            
    def experiment_MLP(self):
        """
        Running all the different MLP classifiers for all devices and prints
        their AUC value accordingly 
        """
        for device in self.devices:
            for first_layer_neurons in [1,2,3,4,5]:
                for second_layer_neurons in [1,2,3,4,5]:
                    clf = self.load_model_from_pkl(r"C:\Users\deanecke\Documents\Project_updated\IoT-device-type-identification-master\models\{0}\{0}_MLP_{1}_{2}_sgd.pkl".format(device,first_layer_neurons,second_layer_neurons))
                    y_test = np.array(pd.Series(self.get_is_dev_vec(device, self.y_test)))
                    pred = np.array(clf.predict(self.x_test))
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
                    print("DEVICE:{0},FIRST LAYER:{1},SECOND LAYER:{2}".format(device, first_layer_neurons, second_layer_neurons))
                    print(metrics.auc(fpr, tpr))


    def load_model_from_pkl(self, pkl_file_full_path):
        return joblib.load(pkl_file_full_path)

    def is_dev(self, this_dev_name, dev_name):
        return 1 if this_dev_name == dev_name else 0
    
    def clear_missing_data(self, df):
        df_with_nan = df.replace("?", np.NaN)
        return df_with_nan.dropna(0)
    
    
    def perform_feature_scaling(self, x_train):
        """
        This method is used in order to perform feature scaling according to the 
        min-max scaler. The scaler can be replaced with another one, like the
        standard scaler 
        """
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        return pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)


    def get_is_dev_vec(self, this_dev_name, dev_names):
        """
        This method generates a list with entries 0 or 1 to indicate which of the
        entries in the dev_names list is the device we are currently training/testing
        a classifier for. 
        """
        return [self.is_dev(this_dev_name, dev_name) for dev_name in dev_names]