# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 21:04:51 2021

@author: user
"""

from abc import ABCMeta, abstractmethod # interface
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class FeatureSelector(metaclass=ABCMeta): # 就像是Java中的interface
    @abstractmethod
    def PrintImportances(self):
        pass
    
    @abstractmethod
    def PlotImportances(self):
        pass
    
    @abstractmethod
    def GetSelectedTrainData(self):
        pass  
    
    @abstractmethod
    def Predict(self):
        pass 
    
    @abstractmethod
    def ConfusionMatrix(self):
        pass 



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
class ExtraTrees(FeatureSelector):  

    def __init__(self, n_estimators, X, y):
        self.extra_trees = ExtraTreesClassifier(n_estimators=n_estimators)
        self.extra_trees = self.extra_trees.fit(X, y)
        self.X = X
        self.y = y       
        self.importances = self.extra_trees.feature_importances_ 
        self.idxs = np.argsort(self.importances)[::-1]
        
    def PrintImportances(self):
        print("Feature ranking:")  
        for f in range(self.X.shape[1]):
            print("\t %d. feature %d: %s (%f)" %
                  (f + 1, self.idxs[f], self.X.columns[self.idxs[f]], self.importances[self.idxs[f]]))
        
    def PlotImportances(self):
        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar( range( self.X.shape[1]), self.importances[self.idxs],
                color="r", align="center")
        plt.xticks(range(self.X.shape[1]), self.idxs)
        plt.xlim([-1, self.X.shape[1]])
        plt.show()
        
    def GetSelectedTrainData(self):
        extra_trees_model = SelectFromModel(self.extra_trees, prefit=True)
        return pd.DataFrame(extra_trees_model.transform(self.X))
    
    def Predict(self, test_X):
        self.y_pred = self.extra_trees.predict(test_X)
    
    def ConfusionMatrix(self, y_true, display_labels):        
        cm = confusion_matrix(y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp = disp.plot(include_values=True, cmap='Blues')
    
    

from sklearn.ensemble import RandomForestClassifier

class RandomForest(FeatureSelector):  

    def __init__(self, n_estimators, max_features, max_depth, min_samples_leaf, X, y):
        self.random_forest = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features, 
                                                  max_depth = max_depth, min_samples_leaf = min_samples_leaf)
        self.random_forest = self.random_forest.fit(X, y)
        self.X = X
        self.y = y       
        self.importances = self.random_forest.feature_importances_    
        self.idxs = np.argsort(self.importances)[::-1]
        
    def PrintImportances(self):
        print("Feature ranking:")    
        for f in range(self.X.shape[1]):
            print("\t %d. feature %d: %s (%f)" %
                  (f + 1, self.idxs[f], self.X.columns[self.idxs[f]], self.importances[self.idxs[f]]))
        
    def PlotImportances(self):
        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar( range( self.X.shape[1]), self.importances[self.idxs],
                color="r", align="center")
        plt.xticks(range(self.X.shape[1]), self.idxs)
        plt.xlim([-1, self.X.shape[1]])
        plt.show()
        
    def GetSelectedTrainData(self):
        random_forest_model = SelectFromModel(self.random_forest, prefit=True)
        return pd.DataFrame(random_forest_model.transform(self.X))
    
    def Predict(self, test_X):
        self.y_pred = self.random_forest.predict(test_X)
    
    def ConfusionMatrix(self, y_true, display_labels):        
        cm = confusion_matrix(y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp = disp.plot(include_values=True, cmap='Blues')
        
               
        
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostedDecisionTrees(FeatureSelector):  

    def __init__(self, n_estimators, learning_rate, max_depth, X, y):
        self.gbdt = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate = learning_rate, 
                                               max_depth = max_depth, random_state=0)
        self.gbdt = self.gbdt.fit(X, y)
        self.X = X
        self.y = y       
        self.importances = self.gbdt.feature_importances_    
        self.idxs = np.argsort(self.importances)[::-1]
        
    def PrintImportances(self):
        print("Feature ranking:")    
        for f in range(self.X.shape[1]):
            print("\t %d. feature %d: %s (%f)" %
                  (f + 1, self.idxs[f], self.X.columns[self.idxs[f]], self.importances[self.idxs[f]]))
        
    def PlotImportances(self):
        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar( range( self.X.shape[1]), self.importances[self.idxs],
                color="r", align="center")
        plt.xticks(range(self.X.shape[1]), self.idxs)
        plt.xlim([-1, self.X.shape[1]])
        plt.show()
        
    def GetSelectedTrainData(self):
        gbdt_model = SelectFromModel(self.gbdt, prefit=True)
        return pd.DataFrame(gbdt_model.transform(self.X))
    
    def Predict(self, test_X):
        self.y_pred = self.gbdt.predict(test_X)
    
    def ConfusionMatrix(self, y_true, display_labels):        
        cm = confusion_matrix(y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp = disp.plot(include_values=True, cmap='Blues')