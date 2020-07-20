# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:12:20 2020

@author: Teuku Hashrul 
"""


import pandas as pd 
import numpy as np 
import shutil
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import AgglomerativeClustering, KMeans
import os
from sklearn.metrics.pairwise import cosine_similarity , cosine_distances
from sklearn.metrics import silhouette_score
import sys
np.set_printoptions(threshold=sys.maxsize)
import plotModule

dataset = pd.read_csv('hasilEksperimen_FinalForm.csv') 

# =============================================================================
# use one hot encoding to make column actors become the features forclustering 
# =============================================================================

#split by , to make actors 
splitted_actors = (dataset.set_index(dataset.columns.drop('actors',1).tolist())
        .actors.str.split(',', expand=True)
        .stack()
        .reset_index()
        .rename(columns={0:'actors'})
        .loc[:, dataset.columns]
        )
#preprocessing removing front space 
splitted_actors['actors'] = splitted_actors['actors'].str.lstrip()
#preprocessing lowercase 
splitted_actors['actors'] = splitted_actors['actors'].str.lower()

# to convert actor into one hot style
onehot_actors = pd.crosstab(splitted_actors['title'],splitted_actors['actors']).rename_axis(None,axis=1).add_prefix('')

merged_inner = pd.merge(left=dataset,right=onehot_actors, left_on='title', right_on='title')

cluster = pd.merge(left=dataset[['title' , 'revenue']],right=onehot_actors, left_on='title', right_on='title')

merged_inner.to_csv('IMDBMOVIE_withonehotactor.csv')
cluster_del  = cluster  
del cluster_del['title']
del cluster_del['revenue']


# fulldata : dataset with all the column 
# features : column used from the fullDataset  (sub dataframe from full Data)
# methods : 'kmeans' or 'agglo' or any other methods 
def generateCluster(fullDataset,features , method , n, titlefeatures):
   
    # iterate every label  
    allLabelMean = pd.DataFrame({"n:"+ str(n): []}) 
    # create dataframe consist of label and the meancluster distance to join with the real dataset 
    labelMeanDistance = pd.DataFrame({'label':[],'intraclusterdistance':[]})
    if method == 'kmeans':
        kmeans = KMeans(n_clusters = n)
        kmeans.fit(features)
        cluster_del['label'] = kmeans.labels_ 
        fullDataset['label'] = kmeans.labels_
        centroid = pd.DataFrame(kmeans.cluster_centers_) 
        #remove label for countable
        centroid = centroid.iloc[:,:-1]
        for index in range(0,n):
            # take the only label in idx: 
            rowLabelNow = cluster_del[cluster_del['label'] == index]
            del rowLabelNow['label']
            #locate the centroid
            centroidNow = centroid.iloc[index] 
       
            #count all the euclidean distance between every row in row labelnow and the centroidNow and put it in the label Now 
            dist = (rowLabelNow - np.array(centroidNow)).pow(2).sum(1).pow(0.5)
            meanDistNow = np.mean(dist) 
            
            allLabelMean = allLabelMean.append({"n:"+str(n):meanDistNow}, ignore_index = True)
            labelMeanDistance = labelMeanDistance.append({'label':index,'intraclusterdistance':meanDistNow}, ignore_index = True) 
    elif method == 'agglo':
        agglo = AgglomerativeClustering(n_clusters = n , affinity='cosine', linkage='average')
        agglo.fit(features)
         
        cluster_del['label'] = agglo.labels_ 
        fullDataset['label'] = agglo.labels_
        
        print('experiment ' + str(n))
        for index in range(0,n):
            # take the only label in idx: 
            rowLabelNow = cluster_del[cluster_del['label'] == index]
            del rowLabelNow['label']
            #create centroid from mean of all the member of the cluster
            
            
            centroidNow = np.mean(rowLabelNow)
            
            if index == 8:
                print(rowLabelNow)
                print(fullDataset[fullDataset['label'] == 8])
                
            #count all the cosine similarity distance between every row in row labelnow and the centroidNow and put it in the label Now 
            #dist = (rowLabelNow - np.array(centroidNow)).pow(2).sum(1).pow(0.5) 
            
            dist = 0
            lenn = 0
            for idx, row in rowLabelNow.iterrows():   
                cosine = cosine_similarity([row],[np.array(centroidNow)])
                dist =  dist + cosine[0][0]
                lenn = lenn + 1
            meanDistNow = dist / lenn
            
            
            allLabelMean = allLabelMean.append({"n:"+str(n):meanDistNow}, ignore_index = True)
            print('label : ' + str(index) + ' mean distance: ' + str(meanDistNow))
            labelMeanDistance = labelMeanDistance.append({'label':index,'intraclusterdistance':meanDistNow}, ignore_index = True) 

    #leftouter join labelMeanDistance and the real dataset clustered 
    dataset_clusteredmean = pd.merge(labelMeanDistance,fullDataset,how='left', on = ['label'])
       
    folder = ''
    if method == 'kmeans':
        folder ='kmeans\\kmeans_n'+str(n)+'\\'
    elif method == 'agglo' : 
        folder = 'agglo\\agglo_n' +str(n)+ '\\'
        
        
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
        
    #save dataset with label and the mean distance 
    dataset_clusteredmean.to_csv(folder+'clustered'+titlefeatures+str(n)+'_dataset.csv')
    # save model using pickle
    model = None
    if method == 'kmeans':
        model = kmeans
    elif method == 'agglo':
        model = agglo
    pickle.dump(model, open(folder+'model'+'_clustered_'+str(n)+titlefeatures, 'wb'))
    return allLabelMean

# =============================================================================
# 
# =============================================================================
    
silhouettescore_ex = pd.DataFrame({"n":[] , "score":[]})
n_experiment = [10,50,100,200,300] 
for i in n_experiment: 
    kmeans = KMeans(n_clusters=i) 
    kmeans.fit(cluster_del) 
    scorek = silhouette_score(cluster_del , kmeans.labels_)
    silhouettescore_ex= silhouettescore_ex.append({"n":str(i), "score" :scorek} , ignore_index = True) 


plotModule.barchart(silhouettescore_ex['n'] , silhouettescore_ex['score'] , 'Number of Clusters' , 'Silhouette Score', 'Kmeans Silhouette Score Actor Comparison,',
                    'kmeansSilhouetescore'+ str(n_experiment[0]) + '-' +str(n_experiment[len(n_experiment)-1])+'.jpg')


    

#also need to count timer 
import timeit 
start = timeit.default_timer()
#switchable 'kmeans' or 'agglo'
method = 'agglo'
allexperimentMean = pd.DataFrame({'n':[],'meanintracluster':[]})
experiment =[10,100,200,300,400]
for n_experiment in experiment:
    allLabelMean = generateCluster(dataset,cluster_del ,method ,n_experiment, 'actor') 
    allexperimentMean = allexperimentMean.append({'n':n_experiment,'meanintracluster':[allLabelMean['n:'+str(n_experiment)]]},ignore_index=True)

stop = timeit.default_timer()
estimatedTime = stop - start   
        
fig,ax = plt.subplots() 
plt.title(method+' intracluster mean distance comparison \n using box and whisker')
plt.ylabel('how many cluster')
plt.xlabel('intracluster mean distribution for every label')
ax.set_yticklabels(allexperimentMean['n'])
ax.boxplot(allexperimentMean['meanintracluster'] , vert=False) 



