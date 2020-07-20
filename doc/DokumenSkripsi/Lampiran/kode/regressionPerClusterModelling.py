# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 07:40:58 2020

@author: Teuku Hashrul 
"""
import pickle
from scipy.stats.stats import pearsonr 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor  
import os 
import shutil
import pandas as pd 
import numpy as np
import plotModule
#experiment to train all the cluster 
# since 160 is better cluster, we assigned to the classificatication 

clustered_dataframe = pd.read_csv('agglo\\agglo_n180\\clusteredactors180_dataset.csv') 
distinct_label = clustered_dataframe[['label']].drop_duplicates()

folderName = 'trained_modelPerCluster'
if os.path.exists(folderName):
    shutil.rmtree(folderName)    
os.makedirs(folderName)
for labelNow in distinct_label.label:
    
    dataLabelNow = clustered_dataframe[clustered_dataframe['label'] == labelNow]
     # create directory for this label for every purpose  
    thisLabelFolderName = 'actor-Clustered-' + str(labelNow) 
    thisLabelDirPath = folderName+'\\'+thisLabelFolderName 
    if os.path.exists(thisLabelDirPath):
        shutil.rmtree(thisLabelDirPath)
    os.makedirs(thisLabelDirPath)
    
     #save the data 
    feature_list = ['runtime', 'rating' , 'votes' , 'metascore', 'us_budget'] 
    
        
     
    name_DataLabelNow = 'actor-'+'label-'+str(labelNow)+'_dataset.csv'
    dataLabelNow.to_csv(thisLabelDirPath+'\\'+name_DataLabelNow) 
    
    response_list = ['revenue', 'roi', 'profit']
    for response in response_list:
        thisResponseLabelDirPath = thisLabelDirPath + '\\' + response
        if os.path.exists(thisResponseLabelDirPath):
            shutil.rmtree(thisResponseLabelDirPath)
        os.makedirs(thisResponseLabelDirPath)

        this_response     = dataLabelNow[[response]] 
        y_arr = np.array(this_response.values)
        
        #create dataframe to store best pearson score 
        pearson_feature_rank = pd.DataFrame({'feature': [] , 'score': []})
        
        for feature in feature_list:
        
            this_feature      = dataLabelNow[[feature]]
            x_arr = np.array(this_feature.values)
            
            # create model
            thisRegressionModel = LinearRegression()
            thisRegressionModel.fit(x_arr , y_arr) 
        
            # save model 
            modelFileName        = 'actor-label-'+str(labelNow)+'-'+feature+'-LinearRegression.sav'
            modelFileNameDirPath = thisResponseLabelDirPath + '\\' + modelFileName
            pickle.dump(thisRegressionModel, open(modelFileNameDirPath, 'wb')) 
        
            polynomial_features= PolynomialFeatures(degree=2)
            x_poly = polynomial_features.fit_transform(x_arr)
    
            thisPolynomModel = LinearRegression()
            thisPolynomModel.fit(x_poly, y_arr)
        
            polynomModelFileName = 'actor-label-'+str(labelNow)+'-'+feature+'-PolynomRegression.sav'
            polynomModelDirPath = thisResponseLabelDirPath + '\\' + polynomModelFileName
            pickle.dump(thisPolynomModel, open(polynomModelDirPath, 'wb'))
        
          
            
            # Scatter plot every feature with the revenue 
            if(dataLabelNow.shape[0] > 1):
                scatterfeature_responseDirFileName = thisResponseLabelDirPath +'\\actor-label-'+str(labelNow)+'-'+feature+'-'+response+'scatterPlot.jpg' 
                plotModule.scatterplot(x_arr , y_arr , feature , response,scatterfeature_responseDirFileName)
                
                # count pearson score 
                pearsonfeatureNow = pearsonr(dataLabelNow[feature] , dataLabelNow[response])[0 ]
                pearson_feature_rank = pearson_feature_rank.append({'feature':feature , 'score':pearsonfeatureNow} , ignore_index = True) 
        
        
        # create multi feature model 
        multiFeature = dataLabelNow[feature_list]
        multi_arr = np.array(multiFeature.values) 
        decTree_Regressor = DecisionTreeRegressor(random_state = 0,max_depth =3)  
        decTree_Regressor.fit(multi_arr , y_arr)
        decTreeRegFileName = 'actor-label-'+str(labelNow)+'-multifeature-DecisionTreeRegressor.sav'
        decTreeRegDirPath = thisResponseLabelDirPath + '\\' + decTreeRegFileName
        pickle.dump(decTree_Regressor , open(decTreeRegDirPath, 'wb')) 
        
    
    
        pearsonFeatureScoreFileName = 'actor-label-'+ str(labelNow)+'pearsonScoreComparison.csv'    
        pearson_feature_rank.to_csv(thisResponseLabelDirPath+'\\'+pearsonFeatureScoreFileName)
            
        
        
        
    
    
    

    
    
    
