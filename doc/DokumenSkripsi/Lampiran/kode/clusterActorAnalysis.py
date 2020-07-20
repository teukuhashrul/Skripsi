# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:50:15 2020

@author: lenovo
"""


import shutil
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import AgglomerativeClustering, KMeans
import os
from sklearn.metrics.pairwise import cosine_similarity , cosine_distances
import sys
np.set_printoptions(threshold=sys.maxsize)
import plotModule
from scipy.stats.stats import pearsonr 
# =============================================================================
# #import plotModuleFromAnotherDirectory (minggu 3) 
# # some_file.py
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, 'D:\backup\Semester 8\Skripsi 2\Minggu 3')
# =============================================================================

# ReadTheClusteredActorDataset 
dataset = pd.read_csv('clusteredactor200_dataset.csv')

#count every label from the clustered data
countEveryLabelMember = dataset[['label' ,'title']].groupby('label').count()

countEveryLabelMember = countEveryLabelMember.sort_values('title' , ascending=False)
#visualize with barchart 
# =============================================================================
# plotModule.barcharth(countEveryLabelMember.index, countEveryLabelMember.title  , 'label' , 'count',
#                      '200 n Agglomerative Clustered Label Member Distribution', '200_NAggloClusteredMemberLabelDistribution.jpg')
# 
# =============================================================================
# ambil 2 kelompok yang anggota actornya mirip dari data 200 clustered actor
possible1 = dataset[dataset['label'] == 1] 
possible2 = dataset[dataset['label'] == 21] 



# ambil salah kelompok 
split_genre = plotModule.split_column_actor(possible1)
genreonly = split_genre['actors']
genreonly = genreonly.astype(str)
np_genre = np.array(genreonly).astype(str)
text_1= " ".join(np_genre).lower()

split_genre = plotModule.split_column_actor(possible2)
genreonly = split_genre['actors']
genreonly = genreonly.astype(str)
np_genre = np.array(genreonly).astype(str)
text_2= " ".join(np_genre).lower()

# visualize clustered actor 
plotModule.generateWordCloud(text_1 , 'Label 1' ,'Actor that featured in the cluster 1 - IntraCluster = 0.58')
plotModule.generateWordCloud(text_2 , 'label 21', 'Actor that featured in the cluster 21 - IntraCluster = 0.51 ')

# visualize the revenue range if we hire the those actor 
plotModule.boxplot(possible1['revenue'] ,False ,'Cluster-1 revenue range' ,'first200cluster-Label1RevenueRange.jpg') 
plotModule.boxplot(possible2['revenue'] ,False ,'Cluster-21 revenue range', 'first200cluster-Label21RevenueRange')

countEveryLabelMember['label_2'] = countEveryLabelMember.index
countEveryLabelMember = countEveryLabelMember.rename(columns={"title":"count"})


# =============================================================================
# Percobaan buang outlier 
# =============================================================================

#take the cluster that have more than one film / removing possible outlier
removedOutlier = countEveryLabelMember[countEveryLabelMember['count']> 1] 

#join the Data with the count 
data_removedOutlier = pd.merge(dataset , removedOutlier, how='right', left_on = 'label' , right_on = 'label_2') 

# copy removed possible outlier 
secondCluster = data_removedOutlier
# removed clusterLabel because we want to cluster again
del secondCluster['label']
del secondCluster['intraclusterdistance'] 
del secondCluster['count']
del secondCluster['label_2'] 



#split by , to make actors 
splitted_actors = (secondCluster.set_index(secondCluster.columns.drop('actors',1).tolist())
        .actors.str.split(',', expand=True)
        .stack()
        .reset_index()
        .rename(columns={0:'actors'})
        .loc[:, secondCluster.columns]
        )
#preprocessing removing front space 
splitted_actors['actors'] = splitted_actors['actors'].str.lstrip()
#preprocessing lowercase 
splitted_actors['actors'] = splitted_actors['actors'].str.lower()

# to convert actor into one hot style
onehot_actors = pd.crosstab(splitted_actors['title'],splitted_actors['actors']).rename_axis(None,axis=1).add_prefix('')

merged_inner = pd.merge(left=secondCluster,right=onehot_actors, left_on='title', right_on='title')

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
# second actor cluster attempt
# =============================================================================

#also need to count timer 
import timeit 
start = timeit.default_timer()
#switchable 'kmeans' or 'agglo'
method = 'agglo'
allexperimentMean = pd.DataFrame({'n':[],'meanintracluster':[]})
experiment =[100,150 ,200 , 250  , 300]
for n_experiment in experiment:
    allLabelMean = generateCluster(secondCluster,cluster_del ,method ,n_experiment, 'actorsecondattempt') 
    allexperimentMean = allexperimentMean.append({'n':n_experiment,'meanintracluster':[allLabelMean['n:'+str(n_experiment)]]},ignore_index=True)

stop = timeit.default_timer()
estimatedTime = stop - start   
        
fig,ax = plt.subplots() 
plt.title(method+' intracluster mean distance comparison \n using box and whisker actor second attempt')
plt.ylabel('how many cluster')
plt.xlabel('intracluster mean distribution for every label')
ax.set_yticklabels(allexperimentMean['n'])
ax.boxplot(allexperimentMean['meanintracluster'] , vert=False) 

# =============================================================================
# March ,1 experiment only run the library 
# =============================================================================


# since we choose 200 as a best optimum cluster actor
# read the generated data  
actorClusteredFixedDataset = pd.read_csv("agglo/agglo_n200/clusteredactorsecondattempt200_dataset.csv") 

# get label only
labelOnly = actorClusteredFixedDataset[['label']] 

# distinct label for looping purpose
labelOnly = labelOnly.drop_duplicates() 

# HYPOTESIS 1 . create dataframe for actor representatives hypothesis
actorContributionPerCluster  = pd.DataFrame({"actorcluster":[], "clustercontribution":[]})  

# HYPOTHESIS 1.1 : there are multiple actor representatives 
multipleActorRepresentativesPerCluster = pd.DataFrame({"cluster_actor_label":[], "howmanyactor":[] , "actornames":[]})


#HYPOTESIS 2. create dataframe -> every actor representatives have their own genre specialty 
actorGenreMostPerCluster = pd.DataFrame({'genrecluster':[] , "clustercontribution":[]}) 


#HYPOTHESIS 3. create dataframe to count their favourite genre revenue contribution 
actorFavGenreRevenuePerCluster = pd.DataFrame({'revenueFavGenre':[], "clustercontribution":[]})


#HYPOTHESIS 3.2 : CREATE DATAFRAME THAT CONSISTENT OR NOT 
actorFavGenre_RevenueConsistencyPerCluster = pd.DataFrame({'cluster_actor_label':[],'actor':[],'favgenre':[], 'favgenrecontribution':[], 'toprevenuegenre':[] , 'toprevenuecontribution':[] , 'status':[]})





#HIPOTHESIS PROFIT 1 : PROFIT FOR EVERY GENRE CONTRIBUTE TO ABOVE 50 PERCENT 
actorFavGenre_ProfitConsistencyPerCluster = pd.DataFrame({'cluster_actor_label':[] , 'actor':[] , 'favgenre':[], 'favgenrecontribution':[]})

# HIPOTHESIS ROI 1   : ROI FOR EVERY GENRE FAVOURITE CONTRIBUTE TO ABOVE 50 PERCENT 
actorFavGenre_RoiConsistencyPerCluster = pd.DataFrame({'cluster_actor_label':[], 'actor':[], 'favgenre':[], 'favgenrecontribution':[]})

##Barchart every cluster mean revenue 
youtubeMeanView_PerCluster = pd.DataFrame({'cluster_actor_label':[], 'mean_youtube_view':[]}) 

#Barchart every cluster pearson revenue profit roi 
youtubeViewPearson_perCluster = pd.DataFrame({'cluster_actor_label':[], 'pearson_revenue':[],'pearson_profit':[],'pearson_roi':[]})

#Barchart every cluster actor hashtag count pearson 
instagramHashtagPearson_perCluster = pd.DataFrame({'cluster_actor_label':[], 'pearson_revenue':[],'pearson_profit':[],'pearson_roi':[]})




# create super directory for saving the actor experiment 
folderName = 'actorClusterAnalysis'  
if os.path.exists(folderName):
    shutil.rmtree(folderName)    
os.makedirs(folderName)
for index,label in labelOnly.iterrows():
    #get label now 
    labelNow = label[0]
    print(labelNow)
    
    # create directory for this label for every purpose  
    thisLabelFolderName = 'actor-Label-' + str(labelNow) 
    thisLabelDirPath = folderName+'\\'+thisLabelFolderName 
    if os.path.exists(thisLabelDirPath):
        shutil.rmtree(thisLabelDirPath)
    os.makedirs(thisLabelDirPath)
    
    #select the data for label Now 
    dataLabelNow = actorClusteredFixedDataset[actorClusteredFixedDataset['label'] == labelNow] 
    #save the data 
    name_DataLabelNow = 'actor-'+'label-'+str(labelNow)+'_dataset.csv'
    dataLabelNow.to_csv(thisLabelDirPath+'\\'+name_DataLabelNow) 
    
    #generate WordCloud Actor from labelNow 
    split_column_actor = plotModule.split_column_actor(dataLabelNow)
    #trim actor cause there are 'kirsten dunst' and ' kirsten dunst' 
    split_column_actor['actors'] = split_column_actor['actors'].str.strip()
    
    actoronly = split_column_actor['actors']
    actoronly = actoronly.astype(str)
    np_actor = np.array(actoronly).astype(str)
    text_actor= " ".join(np_actor).lower() 
    
    wordCloudFileName = thisLabelDirPath + '\\wordcloud-actorlabel-'+str(labelNow)+'.jpg'
    plotModule.generateWordCloud(text_actor , wordCloudFileName ,'Actor Distribution for Label cluster :' + str(labelNow))
    
    
    #boxPlot revenue for this cluster revenue estimation 
    boxplot_revenueName = thisLabelDirPath + '\\boxandwhiskerRevenue-actorlabel-'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['revenue'],False , 'Box and Whisker Revenue Distribution actor cluster-'+str(labelNow) , boxplot_revenueName) 
    
    #boxplot rating for this cluster rating estimation 
    boxplot_ratingName =  thisLabelDirPath + '\\boxandwhiskerRating-actorlabel-'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['rating'],False , 'Box and Whisker Rating Distribution actor cluster-'+str(labelNow) , boxplot_ratingName) 
    
    #boxPlot profit for this cluster estimation 
    boxplot_profitName = thisLabelDirPath + '\\boxandwhisker-Profit-actorlabel'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['profit'],False , 'Box and Whisker Profit Distribution actor cluster-'+str(labelNow), boxplot_profitName) 
    
    #boxplot roi for this cluster estimation 
    boxplot_roiName = thisLabelDirPath + '\\boxandwhisker-Roi-actorlabel'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['roi'], False, 'Box and Whisker Roi Distribution actor cluster-'+str(labelNow), boxplot_roiName)
    
    #boxplot youtubeview for this cluster estimation 
    boxplot_youtubeViewName =  thisLabelDirPath + '\\boxandwhisker-YoutubeView-actorlabel'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['viewCount'], False, 'Box and Whisker YoutubeView Distribution actor cluster-'+str(labelNow), boxplot_youtubeViewName) 
    
    #boxplot instagramhashtagtitle count for this cluster estimation 
    boxplot_instagramhashtagCountName = thisLabelDirPath + '\\boxandwhisker-InstagramHashtagCount-actorlabel'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['hashtagcount'], False, 'Box and Whisker Instagram Hashtag Count Distribution actor cluster-'+str(labelNow), boxplot_youtubeViewName) 
    
    

    
    
    split_genre_actor = plotModule.split_column_genre(dataLabelNow)
    #bar chart for every genre distribution 
    genre_revenue = split_genre_actor[['genre', 'revenue']] 
    meanRevenuePerGenre = genre_revenue.groupby('genre').mean() 
    meanRevenuePerGenre = meanRevenuePerGenre.sort_values('revenue',ascending=True)
    #path to save and visualize
    barchartPerGenreRevenueMean_Name = thisLabelDirPath + '\\barchartMeanGenreRevenue-actorlabel-'+str(labelNow)+'.jpg'
    plotModule.barcharth(meanRevenuePerGenre.index,meanRevenuePerGenre.revenue,'Genre','Mean Revenue','Barchart Mean Revenue Per Genre Distribution actor cluster-'+str(labelNow),barchartPerGenreRevenueMean_Name)
    
    
    #bar chart for every rating distribution 
    genre_rating = split_genre_actor[['genre' , 'rating']] 
    meanRatingPerGenre = genre_rating.groupby('genre').mean()
    meanRatingPerGenre = meanRatingPerGenre.sort_values('rating',ascending=True)
    #path to save and visualize
    barchartPerGenreRatingMean_Name = thisLabelDirPath + '\\barchartMeanGenreRating-actorlabel-'+str(labelNow)+'.jpg'
    plotModule.barcharth(meanRatingPerGenre.index , meanRatingPerGenre.rating, 'Genre','Mean Rating', 'Barchart Mean Rating Per Genre Distribution actor cluster-'+str(labelNow),barchartPerGenreRatingMean_Name) 
    
    
    
    # count every actor play how many title 
    actor_title = split_column_actor[['actors' , 'title']]
    countAllActorInThisCluster = actor_title.groupby('actors').count() 
    # howManyFilmActorPlaysInCluster / howManyTitlesInTheCluster * 100
    countAllActorInThisCluster['contribution'] = countAllActorInThisCluster['title'] / dataLabelNow.shape[0] * 100 
    actorContributionDataFrameName = thisLabelDirPath + '\\cluster-'+str(labelNow)+'-actorTitleContribution.csv'
    #saveThisClusterActorContribution
    countAllActorInThisCluster.to_csv(actorContributionDataFrameName)
    #sort to get most contributing play in this cluster 
    countAllActorInThisCluster = countAllActorInThisCluster.sort_values('contribution', ascending=False) 
    mostPlayedActor = countAllActorInThisCluster.index[0]+'-'+str(labelNow)
    mostPlayedContribution = countAllActorInThisCluster.contribution[0] 
    
    #add to The Global DataFrame 
    actorContributionPerCluster = actorContributionPerCluster.append({"actorcluster":mostPlayedActor, "clustercontribution":mostPlayedContribution} , ignore_index=True)
    
    
    #count every movie 
    genre_title = split_genre_actor[['genre', 'title']] 
    countAllGenreInThisCluster = genre_title.groupby('genre').count() 
    # howManyGenreTitle / allGenreTitle * 100 
    countAllGenreInThisCluster['contribution'] = countAllGenreInThisCluster['title'] / dataLabelNow.shape[0]*100 
    # sort by contribution descending to determine the biggeest actor 
    countAllGenreInThisCluster = countAllGenreInThisCluster.sort_values('contribution' , ascending = False )
    genreContributionDataFrameName = thisLabelDirPath + '\\cluster-'+str(labelNow)+'-genreManyTitleContribution.csv'
    countAllGenreInThisCluster.to_csv(genreContributionDataFrameName) 
    mostPlayedGenreByActor = mostPlayedActor+'-'+countAllGenreInThisCluster.index[0]+'-'+str(labelNow) 
    mostPlayedGenreContribution = countAllActorInThisCluster.contribution[0]
    actorGenreMostPerCluster = actorGenreMostPerCluster.append({'genrecluster':mostPlayedGenreByActor , "clustercontribution":mostPlayedGenreContribution} , ignore_index = True) 
    
    
    # HYPOTHESIS NUMBER 3 
    genre_revenue= split_genre_actor[['genre', 'revenue']] 
    sumAllRevenuePerGenreInThisCluster = genre_revenue.groupby('genre').sum() 
    sumAllRevenuePerGenreInThisCluster['contribution'] = sumAllRevenuePerGenreInThisCluster['revenue'] /dataLabelNow['revenue'].sum() * 100
    sumAllRevenuePerGenreInThisCluster = sumAllRevenuePerGenreInThisCluster.sort_values('contribution', ascending=False) 
    sumActorFavGenreDataFrameName = thisLabelDirPath + '\\cluster-'+str(labelNow)+'-actorFavGenreRevenue.csv'
    sumAllRevenuePerGenreInThisCluster.to_csv(sumActorFavGenreDataFrameName)
    #take the fav genre from hypothesis number two 
    favGenreKey = countAllGenreInThisCluster.index[0] 
    favGenreRevenueContribution = sumAllRevenuePerGenreInThisCluster.contribution[favGenreKey] 
    actorFavGenreRevenueName = mostPlayedActor+favGenreKey
    actorFavGenreRevenuePerCluster = actorFavGenreRevenuePerCluster.append({'revenueFavGenre':actorFavGenreRevenueName, "clustercontribution":favGenreRevenueContribution}, ignore_index= True)
    
    
    #HYPOTHESIS NUMBER 1.2 MULTIPLE ACTORS REPRESENTATIVES
    aboveContributionActorsCount = 0 
    aboveContributionActorNames = ""
    #using hypothesis 1 aggregation how many actor contribute in title 
    multipleAbove50Data = countAllActorInThisCluster[countAllActorInThisCluster['contribution'] >= 50] 
    for index,row in multipleAbove50Data.iterrows():
        #for first index 
        if aboveContributionActorsCount == 0:
            aboveContributionActorNames = aboveContributionActorNames + index
        else:   
            aboveContributionActorNames =  aboveContributionActorNames +  ',' + index 
        aboveContributionActorsCount = aboveContributionActorsCount + 1 
    
    multipleActorRepresentativesPerCluster = multipleActorRepresentativesPerCluster.append({"cluster_actor_label":labelNow, "howmanyactor":aboveContributionActorsCount , "actornames":aboveContributionActorNames} , ignore_index= True)


    #HYPOTHESIS NUMBER 3.2 :  CONSISTENT OR NOT 
    isConsistent = ''
    # fav genre same or not and 
    if sumAllRevenuePerGenreInThisCluster.contribution[favGenreKey] > 50:
        isConsistent = 'consistent'
    else:
        isConsistent = 'not_consistent'
    
    mostPlayedActor = mostPlayedActor
    favGenreKey     = countAllGenreInThisCluster.index[0]
    favGenreContribution = countAllActorInThisCluster.contribution[0]
    
    topRevenueGenre = sumAllRevenuePerGenreInThisCluster.index[0] 
    topRevenueContribution = sumAllRevenuePerGenreInThisCluster.contribution[0]
    
    actorFavGenre_RevenueConsistencyPerCluster = actorFavGenre_RevenueConsistencyPerCluster.append({'cluster_actor_label':labelNow,'actor':mostPlayedActor,'favgenre':favGenreKey, 'favgenrecontribution':favGenreContribution, 'toprevenuegenre':topRevenueGenre , 'toprevenuecontribution':topRevenueContribution , 'status':isConsistent}, ignore_index = True)
    
    # HYPOTHESIS PROFIT : FAVOURITE GENRE contribute more that 50 percent 
    genre_profit = split_genre_actor[['genre', 'profit']]
    sumAllProfitPerGenreInThisCluster = genre_profit.groupby('genre').sum()
    sumAllProfitPerGenreInThisCluster['contribution'] = sumAllProfitPerGenreInThisCluster['profit'] / dataLabelNow['profit'].sum() * 100
    favGenreKey = favGenreKey
    favGenreProfitContribution = sumAllProfitPerGenreInThisCluster.contribution[favGenreKey]
    actorFavGenre_ProfitConsistencyPerCluster = actorFavGenre_ProfitConsistencyPerCluster.append({'cluster_actor_label':labelNow , 'actor':mostPlayedActor, 'favgenre':favGenreKey, 'favgenrecontribution':favGenreProfitContribution}, ignore_index = True)
    
    # HYPOTHESIS ROI : 
    genre_roi = split_genre_actor[['genre', 'roi']]
    sumAllRoiPerGenreInThisCluster = genre_roi.groupby('genre').sum() 
    sumAllRoiPerGenreInThisCluster['contribution'] = sumAllRoiPerGenreInThisCluster['roi'] / dataLabelNow['roi'].sum() * 100
    favGenreRoiContribution = sumAllRoiPerGenreInThisCluster.contribution[favGenreKey]
    actorFavGenre_RoiConsistencyPerCluster = actorFavGenre_RoiConsistencyPerCluster.append({'cluster_actor_label':labelNow, 'actor':mostPlayedActor, 'favgenre':favGenreKey, 'favgenrecontribution':favGenreRoiContribution}, ignore_index = True)

    # BARCHART PEARSON YOUTUBEVIEW and InstagramHashtagCount 
    if dataLabelNow.shape[0] > 1:
        pearson_youtubecount_roi = pearsonr(dataLabelNow['roi'], dataLabelNow['viewCount'])[0]
        pearson_youtubecount_revenue = pearsonr(dataLabelNow['revenue'], dataLabelNow['viewCount'])[0] 
        pearson_youtubecount_profit = pearsonr(dataLabelNow['profit'] , dataLabelNow['viewCount'])[0]
        
        
        pearson_instagramcounthashtag_roi = pearsonr(dataLabelNow['roi'], dataLabelNow['hashtagcount'])[0]
        pearson_instagramcounthashtag_revenue = pearsonr(dataLabelNow['revenue'], dataLabelNow['hashtagcount'])[0]
        pearson_instagramcounthashtag_profit = pearsonr(dataLabelNow['profit'], dataLabelNow['hashtagcount'])[0]
        
        youtubeViewPearson_perCluster = youtubeViewPearson_perCluster.append({'cluster_actor_label':mostPlayedActor, 'pearson_revenue':pearson_youtubecount_revenue
                                                                              ,'pearson_profit':pearson_youtubecount_profit,'pearson_roi':pearson_youtubecount_roi}, ignore_index =True) 
    
        instagramHashtagPearson_perCluster =instagramHashtagPearson_perCluster.append({'cluster_actor_label':mostPlayedActor, 'pearson_revenue':pearson_instagramcounthashtag_revenue
                                                           ,'pearson_profit':pearson_instagramcounthashtag_profit,'pearson_roi':pearson_instagramcounthashtag_roi}, ignore_index = True)


        
    


# saved all The most Played actor representatives 
actorContributionPerCluster.to_csv('allClusterActor-mostPlayed.csv')
actorContributionPerCluster = actorContributionPerCluster.sort_values('clustercontribution',ascending = False)
#visualize 
plotModule.barcharth(actorContributionPerCluster['actorcluster'] , actorContributionPerCluster['clustercontribution'], 'actor representatives' , 'contribution (Percent)' , 'Barchart Every Cluster Most Played Actor', 'barchartMostPlayedActor.jpg')

biggerThan50 = actorContributionPerCluster[actorContributionPerCluster['clustercontribution'] >= 50] 
lowerThan50  = actorContributionPerCluster[actorContributionPerCluster['clustercontribution'] < 50] 

pieChartActorTitleContribution = pd.DataFrame({'label':[], 'count':[]}) 
pieChartActorTitleContribution = pieChartActorTitleContribution.append({'label':'clusterWithActorThat \n PlayedMoreThan50Percent', 'count':biggerThan50.shape[0]}, ignore_index=True)
pieChartActorTitleContribution = pieChartActorTitleContribution.append({'label':'clusterWithActorThat \n PlayedLessThan50Percent', 'count':lowerThan50.shape[0]}, ignore_index=True)
# plot the Disttribution for  comparison
plotModule.pieChart('Actor Representative Cluster Comparison',pieChartActorTitleContribution['label'], pieChartActorTitleContribution['count'])
    
       

#HYPOTHESISNUMBERTWO : EVERY ACTOR HAVE THEIR OWN FAVOURITE GENRE 
actorGenreMostPerCluster.to_csv('allClusterActor-mostFavGenre.csv') 
actorGenreMostPerCluster = actorGenreMostPerCluster.sort_values('clustercontribution', ascending = False) 
#visualize 
plotModule.barcharth(actorGenreMostPerCluster['genrecluster'] , actorGenreMostPerCluster['clustercontribution'],'Most Favourite actor genre' , 'genre contribution in cluster (percent)' , 'Barchart Every Cluster Favourite Genre','barchartMostFavGenreInCluster.jpg')

biggerGenreThan50 = actorGenreMostPerCluster[actorGenreMostPerCluster['clustercontribution'] >= 50] 
lowerGenreThan50  = actorGenreMostPerCluster[actorGenreMostPerCluster['clustercontribution'] < 50]  

pieChartFavGenreTitleContribution = pd.DataFrame({'label':[] , 'count':[]})
pieChartFavGenreTitleContribution = pieChartFavGenreTitleContribution.append({'label':'cluster with most played genre \n more than 50 percent', 'count':biggerGenreThan50.shape[0]} , ignore_index = True)
pieChartFavGenreTitleContribution = pieChartFavGenreTitleContribution.append({'label':'cluster with most played genre \n less than 50 percent', 'count':lowerGenreThan50.shape[0]}, ignore_index=True)
plotModule.pieChart('Actor Possible Favourite Genre \n Contribution perCluster Comparison',pieChartFavGenreTitleContribution['label'], pieChartFavGenreTitleContribution['count'])


#HYPOTHESISNUMBERTHREE : DO THEIR FAVOURITE GENRE MOST CONTRIBUTED TO THEIR REVENUE
actorFavGenreRevenuePerCluster.to_csv('allClusterActor-FavGenreRevenue.csv') 
actorFavGenreRevenuePerCluster = actorFavGenreRevenuePerCluster.sort_values('clustercontribution' , ascending = False)

plotModule.barcharth(actorFavGenreRevenuePerCluster['revenueFavGenre'],actorFavGenreRevenuePerCluster['clustercontribution'], 'cluster-actor-favgenre' , 'Revenue Contribution \n Per Cluster (Percent)', 'BarChart Every Cluster Most Fav Genre \n Revenue Contribution', 'barchartFavGenreRevenueInCluster.jpg') 
pieChartFavGenreRevenueContribution = pd.DataFrame({'label':[] , 'count':[]})  

biggerRevenueThan50 = actorFavGenreRevenuePerCluster[actorFavGenreRevenuePerCluster['clustercontribution'] >= 50] 
lowerRevenueThan50  = actorFavGenreRevenuePerCluster[actorFavGenreRevenuePerCluster['clustercontribution'] <  50]  

pieChartFavGenreRevenueContribution = pieChartFavGenreRevenueContribution.append({'label':'cluster with favourite genre contribution \n more than 50 percent', 'count':biggerRevenueThan50.shape[0]}, ignore_index=True)  
pieChartFavGenreRevenueContribution = pieChartFavGenreRevenueContribution.append({'label':'cluster with favourite genre contribution \n less than 50 percent', 'count':lowerRevenueThan50.shape[0]} , ignore_index=True)
plotModule.pieChart('Actor Favourite Genre Revenue Contribution Comparison', pieChartFavGenreRevenueContribution['label'],pieChartFavGenreRevenueContribution['count'])
 
#HYPOTHESIS NUMBER 1.2 : NOT ONLY LEAD ACTOR BUT A CLUSTER REPRESENTS A PAIR OF ACTOR POSSIBLE ACTOR THAT CONTRIBUTES ABOVE 50% OF THE FILM
multipleActorRepresentativesPerCluster.to_csv('allCluster-possiblePairActorRepresentatives.csv') 
multipleActorRepresentativesCountGroupBy = multipleActorRepresentativesPerCluster[['cluster_actor_label', 'howmanyactor']].groupby('howmanyactor').count() 
multipleActorRepresentativesCountGroupBy['howmanyactor'] = multipleActorRepresentativesCountGroupBy.index
multipleActorRepresentativesCountGroupBy['howmanyactor'] = 'many lead actor : ' + multipleActorRepresentativesCountGroupBy['howmanyactor'].astype(str) 
# create pieChart  
labels = np.array(multipleActorRepresentativesCountGroupBy['howmanyactor']) 
sizes  = np.array(multipleActorRepresentativesCountGroupBy['cluster_actor_label']) 
plt.figure(figsize=(6,6))
plt.title('Every Actor Cluster many \n Actor Representatives', fontweight = 'bold')
patches, texts = plt.pie(sizes , startangle = 90) 
plt.legend(patches , labels , loc='lower left')
plt.axis('equal') 
plt.tight_layout()
plt.show() 

# plot pieChart yang consistent dan tidak 
actorFavGenre_RevenueConsistencyPerCluster.to_csv('allActorCluster-favGenre-Revenue-ConsistentcyDataFrame.csv')
consistentGroupBy = actorFavGenre_RevenueConsistencyPerCluster[['actor', 'status']].groupby('status').count() 
plotModule.pieChart('Actor Cluster Consistent Comparison', consistentGroupBy.index,consistentGroupBy['actor'])


# HYPOTHESIS PROFIT : ACTOR FAV GENRE PROFIT CONTRIBUTE HIGHER THAN 50 PERCENT 
actorFavGenre_ProfitConsistencyPerCluster.to_csv('allClusterActor-FavGenreProfitContribution.csv')
pieChartFavGenreProfitContribution = pd.DataFrame({'label':[], 'count':[]})
higherThan50 = actorFavGenre_ProfitConsistencyPerCluster[actorFavGenre_ProfitConsistencyPerCluster['favgenrecontribution'] >= 50] 
lowerThan50  = actorFavGenre_ProfitConsistencyPerCluster[actorFavGenre_ProfitConsistencyPerCluster['favgenrecontribution'] <  50] 

pieChartFavGenreProfitContribution = pieChartFavGenreProfitContribution.append({'label':'cluster with actor favourite genre \n profit contribution more than 50 percent', 'count': higherThan50.shape[0]}, ignore_index= True)
pieChartFavGenreProfitContribution = pieChartFavGenreProfitContribution.append({'label':'cluster with actor favourite genre \n profit contribution less than 50 percent', 'count': lowerThan50.shape[0]}, ignore_index= True)

plotModule.pieChart('Actor Favourite Genre Profit Contribution Comparison', pieChartFavGenreProfitContribution['label'] , pieChartFavGenreProfitContribution['count']) 

# HYPOTHESIS ROI : ACTOR FAV GENRE ROI CONTRIBUTE HIGHER THAN 50 PERCENT
actorFavGenre_RoiConsistencyPerCluster.to_csv('allClusterActor-FavGenreRoiContribution.csv')
pieChartFavGenreRoiContribution = pd.DataFrame({'label':[], 'count':[]}) 
higherThan50 = actorFavGenre_RoiConsistencyPerCluster[actorFavGenre_RoiConsistencyPerCluster['favgenrecontribution'] >= 50] 
lowerThan50  = actorFavGenre_RoiConsistencyPerCluster[actorFavGenre_RoiConsistencyPerCluster['favgenrecontribution'] < 50]

pieChartFavGenreRoiContribution = pieChartFavGenreRoiContribution.append({'label':'cluster with actor favourite genre \n roi contribution more than 50 percent', 'count': higherThan50.shape[0]}, ignore_index = True)
pieChartFavGenreRoiContribution = pieChartFavGenreRoiContribution.append({'label':'cluster with actor favourite genre \n roi contribution more less 50 percent', 'count': lowerThan50.shape[0]}, ignore_index = True)
plotModule.pieChart('Actor Favourite Genre Roi Contribution Comparison' , pieChartFavGenreRoiContribution['label'], pieChartFavGenreRoiContribution['count'])



# youtube Trailer view to Other REsponse every cluster 
youtubeViewPearson_perCluster = youtubeViewPearson_perCluster.sort_values('pearson_revenue', ascending = False)
youtubeViewPearson_perCluster.to_csv('pearsonAllClusterYoutubeViewandAllResponseDataset.csv')
plotModule.barcharth(youtubeViewPearson_perCluster['cluster_actor_label'],
                     youtubeViewPearson_perCluster['pearson_revenue'],
                     'actor Label', 'Pearson score',
                     'Pearson Score Barchart Every \n Cluster Youtube Views and Revenue', 'pearsonAllClusterYoutubeViewandRevenue.jpg')

instagramHashtagPearson_perCluster = instagramHashtagPearson_perCluster.sort_values('pearson_revenue', ascending = False)
instagramHashtagPearson_perCluster.to_csv('pearsonAllClusterInstagramHashtagCountandAllResponseDataset.csv')
plotModule.barcharth(instagramHashtagPearson_perCluster['cluster_actor_label'],
                     instagramHashtagPearson_perCluster['pearson_revenue'], 
                     'actor Label', 'pearson Score',
                     'Pearson Score Barchart Every \n Cluster Instagram Hashtag Count and revenue', 'pearsonAllClusterInstagramHashtagCountandRevenue.jpg')