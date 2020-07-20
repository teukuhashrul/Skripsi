# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 08:38:09 2020

@author: Teuku Hashrul 
"""


import pandas as pd 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import plotModule
import os 
import shutil

test_Data = pd.read_csv('imdb_finalTestDataSet.csv')
train_Data = pd.read_csv('imdb_finalTrainingDataSet.csv') 

clustered_data = pd.read_csv('agglo\\agglo_n180\\clusteredactors180_dataset.csv')
clustered_data_labelOnly = clustered_data[['title' , 'label']]

merged_trainData = pd.merge(train_Data,clustered_data_labelOnly,how='left' , on='title')

deleting_train = ['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1', 'title', 'genre', 'description',
       'director', 'actors', 'year', 'runtime', 'rating', 'votes' , 
       'revenue' ,'metascore' , 'budget' , 'us_budget' , 'label', 'profit', 'roi']
deleting_test =  ['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1', 'title', 'genre', 'description',
       'director', 'actors', 'year', 'runtime', 'rating', 'votes' , 
       'revenue' ,'metascore' , 'budget' , 'us_budget' , 'profit', 'roi']

# =============================================================================
# CLASSIFICATION ATTEMPT TO FIND GOOD CLUSTER FOR EVERY TEST DATA 
# =============================================================================

train_actorPredictor = merged_trainData.drop(deleting_train , axis =1)
train_response       = merged_trainData[['label']] 

test_actorPredictor = test_Data.drop(deleting_test , axis=1) 
kNN_model_iris = KNeighborsClassifier( n_neighbors = 5, metric = 'euclidean')
 
 #Train the model using the training sets
kNN_model_iris.fit(train_actorPredictor , train_response)
 
#Predict the response for test dataset
y_pred = kNN_model_iris.predict(test_actorPredictor) 
 
test_Data['label'] = y_pred


response_list=['revenue' , 'profit', 'roi']
# result -> 3 response -> 
resultFolder = 'result'
if os.path.exists(resultFolder):
    shutil.rmtree(resultFolder)    
os.makedirs(resultFolder)
for response in response_list:
    thisResponseFolderName = resultFolder + '\\' + response
    if os.path.exists(thisResponseFolderName):
        shutil.rmtree(thisResponseFolderName)    
    os.makedirs(thisResponseFolderName)
    # =============================================================================
    #  REGRESSION ATTEMPT ON EVERY TEST DATA 
    # =============================================================================
    predictionResult_VotesDataFrame = pd.DataFrame({"title":[] , "votes":[], response:[] , response+"_predicted_linear":[] 
    ,"intercept":[], "coef":[], "label":[], response+"_predicted_polynom":[]}) 
    
    predictionResult_MetascoreDataFrame = pd.DataFrame({"title":[] , "metascore":[], response:[] , response+"_predicted_linear":[] 
    ,"intercept":[], "coef":[], "label":[], response+"_predicted_polynom":[]}) 
    
    predictionResult_RatingDataFrame = pd.DataFrame({"title":[] , "rating":[], response:[] , response+"_predicted_linear":[] 
    ,"intercept":[], "coef":[], "label":[], response+"_predicted_polynom":[]}) 
    
    predictionResult_RuntimeDataFrame = pd.DataFrame({"title":[] , "runtime":[], response:[] , response+"_predicted_linear":[] 
    ,"intercept":[], "coef":[], "label":[], response+"_predicted_polynom":[]}) 
     
    predictionResult_BestDataFrame = pd.DataFrame({"title":[] , "feature":[],"value":[] ,"revenue":[] , response+"_predicted_linear":[] 
    ,"intercept":[], "coef":[], "label":[], response+"_predicted_polynom":[]}) 
    
    predictionResult_DecTreeReg = pd.DataFrame({"title":[] , "revenue":[] , response+"_predicted":[], "label":[]})
    
    test_AllIndex = test_Data.index.values
    for indexNow in test_AllIndex: 
        
        print("predicting : Data" + str(indexNow)+ 'for response : ' + response)
        test_DataNow = test_Data.index[indexNow]
        
        movieTitleNow = test_Data.title[indexNow] 
        # real revenue
        movieResponseNow  = test_Data[response][indexNow]
        movieLabelNow = test_Data.label[indexNow] 
        #predictor features
        movieBudgetNow = test_Data.us_budget[indexNow]
        movieVotesNow = test_Data.votes[indexNow] 
        movieMetascoreNow = test_Data.metascore[indexNow]
        movieRatingNow = test_Data.rating[indexNow] 
        movieRuntimeNow = test_Data.runtime[indexNow] 
        
        #FOR LINEAR REGRESSION
        # Y = a + b (feature)
        # a= intercept_
        # b = coef_
    # =============================================================================
    #VOTES ATTEMPT
    # =============================================================================
        # take the model from the loaded model
        dirFileVotesName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-votes' + "-LinearRegression.sav"
        loaded_linearRegressionVotesModel = pickle.load(open(dirFileVotesName, 'rb'))
        intercept_ = loaded_linearRegressionVotesModel.intercept_[0]
        coef_      = loaded_linearRegressionVotesModel.coef_[0,0]    
        #predict attempt
        moviePredictedResponseNow = loaded_linearRegressionVotesModel.predict([[movieVotesNow]])[0,0] 
         
        ## read polynomila features 
        dirPolynomVotesFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-votes'+"-PolynomRegression.sav"
        loaded_polynomRegressionModel = pickle.load(open(dirPolynomVotesFileName, 'rb'))
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform([[movieVotesNow]])
        y_poly_pred = loaded_polynomRegressionModel.predict(x_poly)[0,0]
    
        
        predictionResult_VotesDataFrame = predictionResult_VotesDataFrame.append({"title":movieTitleNow , "votes":movieVotesNow, response:movieResponseNow, response+"_predicted_linear":moviePredictedResponseNow 
        ,"intercept":intercept_, "coef":coef_, "label":movieLabelNow, response+"_predicted_polynom":y_poly_pred} , ignore_index = True) 
    
    # =============================================================================
    #  METASCORE ATTEMPT
    # =============================================================================
        # take the model from the loaded model
        dirFileMetascoreFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-metascore' + "-LinearRegression.sav"
        loaded_linearRegressionMetascoreModel = pickle.load(open(dirFileMetascoreFileName, 'rb'))
        intercept_ = loaded_linearRegressionMetascoreModel.intercept_[0]
        coef_      = loaded_linearRegressionMetascoreModel.coef_[0,0]    
        #predict attempt
        moviePredictedResponseNow = loaded_linearRegressionMetascoreModel.predict([[movieMetascoreNow]])[0,0] 
         
        ## read polynomila features 
        dirPolynomMetascoreFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-metascore'+"-PolynomRegression.sav"
        loaded_polynomRegressionModel = pickle.load(open(dirPolynomMetascoreFileName, 'rb'))
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform([[movieMetascoreNow]])
        y_poly_pred = loaded_polynomRegressionModel.predict(x_poly)[0,0]
    
         
        predictionResult_MetascoreDataFrame = predictionResult_MetascoreDataFrame.append({"title":movieTitleNow , "metascore":movieMetascoreNow, response:movieResponseNow, response+"_predicted_linear":moviePredictedResponseNow
         ,"intercept":intercept_, "coef":coef_, "label":movieLabelNow, response+"_predicted_polynom":y_poly_pred}, ignore_index = True) 
    
        
    
    # =============================================================================
    # RATING ATTEMPT
    # =============================================================================
          # take the model from the loaded model
        dirFileRatingFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-rating' + "-LinearRegression.sav"
        loaded_linearRegressionRatingModel = pickle.load(open(dirFileRatingFileName, 'rb'))
        intercept_ = loaded_linearRegressionRatingModel.intercept_[0]
        coef_      = loaded_linearRegressionRatingModel.coef_[0,0]    
        #predict attempt
        moviePredictedResponseNow = loaded_linearRegressionRatingModel.predict([[movieRatingNow]])[0,0] 
         
        ## read polynomila features 
        dirPolynomRatingFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-rating'+"-PolynomRegression.sav"
        loaded_polynomRegressionModel = pickle.load(open(dirPolynomRatingFileName, 'rb'))
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform([[movieRatingNow]])
        y_poly_pred = loaded_polynomRegressionModel.predict(x_poly)[0,0]
    
        
        predictionResult_RatingDataFrame = predictionResult_RatingDataFrame.append({"title":movieTitleNow , "rating":movieRatingNow, response:movieResponseNow, response+"_predicted_linear":moviePredictedResponseNow
        ,"intercept":intercept_, "coef":coef_, "label":movieLabelNow, response+"_predicted_polynom":y_poly_pred} , ignore_index = True) 
    # =============================================================================
    # RUNTIME ATTEMPT
    # =============================================================================
        
        # take the model from the loaded model
        dirFileRuntimeFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-runtime' + "-LinearRegression.sav"
        loaded_linearRegressionRuntimeModel = pickle.load(open(dirFileRuntimeFileName, 'rb'))
        intercept_ = loaded_linearRegressionRuntimeModel.intercept_[0]
        coef_      = loaded_linearRegressionRuntimeModel.coef_[0,0]    
        #predict attempt
        moviePredictedResponseNow = loaded_linearRegressionMetascoreModel.predict([[movieRuntimeNow]])[0,0] 
         
        ## read polynomila features 
        dirPolynomRuntimeFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-runtime'+"-PolynomRegression.sav"
        loaded_polynomRegressionModel = pickle.load(open(dirPolynomRuntimeFileName, 'rb'))
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform([[movieRuntimeNow]])
        y_poly_pred = loaded_polynomRegressionModel.predict(x_poly)[0,0]
    
        
        predictionResult_RuntimeDataFrame = predictionResult_RuntimeDataFrame.append({"title":movieTitleNow , "runtime":movieRuntimeNow, response:movieResponseNow, response+"_predicted_linear":moviePredictedResponseNow
        ,"intercept":intercept_, "coef":coef_, "label":movieLabelNow, response+"_predicted_polynom":y_poly_pred} , ignore_index = True) 
        
        
    # =============================================================================
    # DECISION TREE MULTI REGRESSOR 
    # =============================================================================
        
         # take the model from the loaded model
        dirFileMultiDecTreeFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-multifeature' + "-DecisionTreeRegressor.sav"
        loaded_DecisionTreeRegressionMultiModel = pickle.load(open(dirFileMultiDecTreeFileName, 'rb'))
        feature_list = ['runtime', 'rating' , 'votes' , 'metascore', 'us_budget'] 
        
        multiFeature = [movieRuntimeNow , movieRatingNow , movieVotesNow , movieMetascoreNow , movieBudgetNow]
        #predict attempt
        moviePredictedResponseNow = loaded_DecisionTreeRegressionMultiModel.predict([multiFeature])
        predictionResult_DecTreeReg = predictionResult_DecTreeReg.append({"title":movieTitleNow , response:movieResponseNow , response+"_predicted":moviePredictedResponseNow, "label":movieLabelNow} , ignore_index = True) 
        
    # =============================================================================
    #  BEST FEATURE COMBINED
    # =============================================================================
        # untuk tiap test data 
        # baca label kelompoknya 
        # baca dataframe best feature nya 
        # ambil urutan satu nama featurenya 
        # terus ambil model yangfeaturenya sama juga
        # yaudah bikin linear sama polynom nya 
        pearsonFileDir   = 'trained_modelPerCluster\\actor-Clustered-'+ str(movieLabelNow)+"\\"+response +'\\actor-label-'+str(movieLabelNow)+'pearsonScoreComparison.csv'
        pearsonDataFrame = pd.read_csv(pearsonFileDir)
        pearsonDataFrame = pearsonDataFrame.sort_values('score' , ascending = False)
        thisLabelBestFeature = pearsonDataFrame.iloc[0].feature
       
        
        bestLinearModelFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-'+ thisLabelBestFeature +"-LinearRegression.sav"
        bestModel_Linear = pickle.load(open(bestLinearModelFileName, 'rb'))
        best_x  = test_Data[thisLabelBestFeature][indexNow]
        bestLinear_predicted = bestModel_Linear.predict([[best_x]])[0,0]
        intercept_ = bestModel_Linear.intercept_[0]
        coef_      = bestModel_Linear.coef_[0,0]    
        
        
        bestPolynomModelFileName = "trained_modelPerCluster\\actor-Clustered-"+str(movieLabelNow)+"\\"+response +"\\actor-label-"+str(movieLabelNow)+'-'+ thisLabelBestFeature +"-PolynomRegression.sav"
        bestModel_Polynom = pickle.load(open(bestPolynomModelFileName, 'rb')) 
        x_poly = polynomial_features.fit_transform([[best_x]])
        bestPolynom_predicted = bestModel_Polynom.predict(x_poly)[0,0] 
        
        predictionResult_BestDataFrame = predictionResult_BestDataFrame.append({"title":movieTitleNow , "feature":thisLabelBestFeature,"value":best_x ,response:test_Data[response][indexNow] , response+"_predicted_linear":bestLinear_predicted 
        ,"intercept":intercept_, "coef":coef_, "label":movieLabelNow, response+"_predicted_polynom":bestPolynom_predicted} , ignore_index = True) 
        
    
    # save dataframe    
    
    predictionResult_VotesDataFrame.to_csv(thisResponseFolderName+'\\'+'predictionResult_VotesOnlyClusteredActor.csv')
    
    predictionResult_MetascoreDataFrame.to_csv(thisResponseFolderName+'\\'+'predictionResult_MetascoreOnlyClusteredActor.csv') 
    
    predictionResult_RatingDataFrame.to_csv(thisResponseFolderName+'\\'+'predictionResult_RatingOnlyClusteredActor.csv') 
    
    predictionResult_RuntimeDataFrame.to_csv(thisResponseFolderName+'\\'+'predictionResult_RuntimeOnlyClusteredActor.csv') 
    
    predictionResult_DecTreeReg.to_csv(thisResponseFolderName+'\\'+'predictionResult_DecTreeRegressorClusteredActor.csv')
    
    predictionResult_BestDataFrame.to_csv(thisResponseFolderName+'\\'+'predictionResult_BestFeatureClusteredActor.csv')
    
    
    # Plot 1 box and whisker 1 feature 
    # -> votes
    # -> metascore
    # -> rating
    # -> runtime
    # -> best 
    # -> DecisionTree 
    # -> 
    arrPredictionResult = [predictionResult_VotesDataFrame,
                           predictionResult_RuntimeDataFrame,
                           predictionResult_RatingDataFrame,
                           predictionResult_MetascoreDataFrame,
                           predictionResult_BestDataFrame] 
    
    featureDirName = thisResponseFolderName+'\\'+'features'
    if os.path.exists(featureDirName):
        shutil.rmtree(featureDirName)    
    os.makedirs(featureDirName)
    for predictionResult in arrPredictionResult:
        # count r2 for linear 
        allPredictedLabel = predictionResult[['label']]
        allPredictedLabel = allPredictedLabel.drop_duplicates('label').label
        
        thisPredictionResult_PredictorName = predictionResult.keys()[1]
        thisPredictedResultR2_DataFrameLinear = pd.DataFrame({'label':[], 'r2':[]})
        thisPredictedResultR2_DataFramePolynom = pd.DataFrame({'label':[] , 'r2':[]})
        for labelNow in allPredictedLabel:
            #count r2 linear for this label in this feature prediction result 
            thisLabelPredictedData = predictionResult[predictionResult['label'] == labelNow]
            r2_linear = r2_score(thisLabelPredictedData[response] , thisLabelPredictedData[response+'_predicted_linear'])
            thisPredictedResultR2_DataFrameLinear = thisPredictedResultR2_DataFrameLinear.append({"label":labelNow , "r2":r2_linear} , ignore_index=True) 
            
            r2_polynom = r2_score(thisLabelPredictedData[response] , thisLabelPredictedData[response+'_predicted_polynom'])
            thisPredictedResultR2_DataFramePolynom = thisPredictedResultR2_DataFramePolynom.append({"label":labelNow , "r2":r2_linear} , ignore_index=True) 
        
        
        # since r2 cannot be count if there are less than 2 observation for every cluster , it will return nan so dropped it 
        thisPredictedResultR2_DataFrameLinear = thisPredictedResultR2_DataFrameLinear.dropna()
        thisPredictedResultR2_DataFramePolynom = thisPredictedResultR2_DataFramePolynom.dropna()
        
        
        
        
        #create multi box and whisker : 1 box and whisker for 1 feature linear and polynom 
        boxWhiskerThisPredictionFeature = pd.DataFrame({'model':[] , 'arrayR2': []}) 
        boxWhiskerThisPredictionFeature = boxWhiskerThisPredictionFeature.append({'model':'linear', 'arrayR2' :thisPredictedResultR2_DataFrameLinear.r2}, ignore_index = True)
        boxWhiskerThisPredictionFeature = boxWhiskerThisPredictionFeature.append({'model':'polynom', 'arrayR2':thisPredictedResultR2_DataFramePolynom.r2} , ignore_index = True)
        
        
        
        #create feature dir Name 
        thisFeatureDirName = featureDirName + '\\' + thisPredictionResult_PredictorName
        if os.path.exists(thisFeatureDirName):
            shutil.rmtree(thisFeatureDirName)    
        os.makedirs(thisFeatureDirName)
        
        thisFeatureResponseMultiBoxPlotDirName = thisFeatureDirName + '\\' + thisPredictionResult_PredictorName+'_multiBoxPlot_'+response+'_r2Score.jpg'
        plotModule.saveMultiBoxPlot(response+'  prediction with ' + thisPredictionResult_PredictorName + ' r2 ' + response +' score box and whisker \n for every cluster using different model ', 
                                False,
                                boxWhiskerThisPredictionFeature.model,
                                'Regressor',
                                'r2 Distribution',
                                boxWhiskerThisPredictionFeature.arrayR2,
                                thisFeatureResponseMultiBoxPlotDirName)
        
        thisFeatureLinearDataFrameName = thisPredictionResult_PredictorName + '_LinearR2DataFrame.csv'
        thisPredictedResultR2_DataFrameLinear.to_csv(thisFeatureDirName + '\\' + thisFeatureLinearDataFrameName) 
        
        thisFeaturePolynomDataFrameName = thisPredictionResult_PredictorName + '_PolynomR2DataFrame.csv'
        thisPredictedResultR2_DataFramePolynom.to_csv(thisFeatureDirName + '\\' + thisFeaturePolynomDataFrameName)
        
            
       
        
        print('----------------------------------------')
    
    # =============================================================================
    #  DECISION TREE REGRESSOR 
    # =============================================================================
    # CREATE FOR DECISION TREE REGRESSOR 
    thisPredictedResultR2_DecTreeDataFrame = pd.DataFrame({'label':[], 'r2':[]})
    
    
    allPredictedLabel = predictionResult_DecTreeReg[['label']]
    allPredictedLabel = allPredictedLabel.drop_duplicates('label').label
    
    for labelNow in allPredictedLabel:
        #count r2 linear for this label in this feature prediction result 
        thisLabelPredictedData = predictionResult_DecTreeReg[predictionResult_DecTreeReg['label'] == labelNow]
        r2_labelNow = r2_score(thisLabelPredictedData[response] , thisLabelPredictedData[response+'_predicted'])
        
        thisPredictedResultR2_DecTreeDataFrame = thisPredictedResultR2_DecTreeDataFrame.append({'label': labelNow , 'r2':r2_labelNow}, ignore_index = True)
        
    # since r2 cannot be count if there are less than 2 observation for every cluster , it will return nan so dropped it 
    thisPredictedResultR2_DecTreeDataFrame = thisPredictedResultR2_DecTreeDataFrame.dropna()
        
        
        
    #create multi box and whisker : 1 box and whisker for 1 feature linear and polynom 
    regTreeR2DirPath = featureDirName+'\\decTree'
    if os.path.exists(regTreeR2DirPath):
        shutil.rmtree(regTreeR2DirPath)    
    os.makedirs(regTreeR2DirPath)
    
    regTreeR2DirPathR2FileName = regTreeR2DirPath + '\\'+ 'DecisionTreeRegressorR2Distribution.csv'
    thisPredictedResultR2_DecTreeDataFrame.to_csv(regTreeR2DirPathR2FileName)
    
    regTreeR2DirPathBoxplot   = regTreeR2DirPath + '\\' + 'boxPlotR2DecisionTreeRegressor.jpg'
    thisPredictedResultR2_DecTreeDataFrame
    plotModule.boxplot(thisPredictedResultR2_DecTreeDataFrame.r2,
                       False, 
                       'Decision Tree Regressor Multi Feature \n R2 Score Distribution',
                       regTreeR2DirPathBoxplot)
        