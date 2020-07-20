# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:25:28 2020

@author: lenovo
"""

import pandas as pd 
import plotModule
import os
import shutil
from scipy.stats.stats import pearsonr 
import numpy as np
# from file genreCluster attempt, we choose that 189 is the optimal cluster with kmeans 

dataframe = pd.read_csv('agglo\\agglo_n189\\clusteredgenre189_dataset.csv')

# how many genre combination 
distinctGenreCombination = dataframe[['genre']].drop_duplicates()

# Global analysis
genre_countTitle = dataframe[['genre' , 'title']].groupby('genre').count()
genre_countTitle  = genre_countTitle.sort_values('title' , ascending = False)

top20_genre_countTitle = genre_countTitle.head(20)
plotModule.barcharth(genre_countTitle.index , genre_countTitle['title'] , 'Genre combination' , 'How Many Films Created','Films created for every genre \n comparison', 'genrecluster_combinationtitle_barchart.jpg')
plotModule.barcharth(top20_genre_countTitle.index , top20_genre_countTitle['title'] , 'Genre combination' , 'Top 20 How Many Films Created','Films created for every genre \n comparison', 'top20genrecluster_combinationtitle_barchart.jpg') 


genre_sumRevenue = dataframe[['genre' , 'revenue']].groupby('genre').sum() 
genre_sumRevenue = genre_sumRevenue.sort_values('revenue' , ascending = False) 
top20_genre_sumRevenue = genre_sumRevenue.head(20)
plotModule.barcharth(genre_sumRevenue.index , genre_sumRevenue['revenue'] ,'Genre Combination' ,'Total Revenue','Total Revenue every genre \n combination barchart ', 'genrecluster_totalRevenue_barchart.jpg')
plotModule.barcharth(top20_genre_sumRevenue.index , top20_genre_sumRevenue['revenue'] ,'Genre Combination' ,'Total Revenue','Total Revenue every genre \n combination barchart ', 'top20_genrecluster_totalRevenue_barchart.jpg')


# get label only
labelOnly = dataframe[['label']] 

# distinct label for looping purpose
labelOnly = labelOnly.drop_duplicates()  

# HYPOTHESIS GENRE.1 : suatu Kombinasi punya actor dominan -> jumlah kontribusi
actorContributionPerCluster = pd.DataFrame({'genrecluster':[],'clustercontribution':[]}) 


# HYPOTHESIS GENRE 2 : actor dominan dari kombinasi punya kontribusi revenue tinggi 
actorRevenueContributionPerCluster = pd.DataFrame({'genrecluster':[], 'clustercontribution':[]}) 

# HYPOTHESIS GENRE PROFIT : actor dominan dari kombinasi punya kontribusi profit tinggi 
actorProfitContributionPerCluster = pd.DataFrame({'genrecluster':[], 'clustercontribution':[]}) 

#HYPOTHESIS GENRE ROI : actor dominan dari kombinasi memiliki kontribusi roi tinggi
actorRoiContributionPerCluster = pd.DataFrame({'genrecluster':[], 'clustercontribution':[]}) 

#Barchart every cluster pearson revenue profit roi 
youtubeViewPearson_perCluster = pd.DataFrame({'genrecluster':[], 'pearson_revenue':[],'pearson_profit':[],'pearson_roi':[]})

#Barchart every cluster actor hashtag count pearson 
instagramHashtagPearson_perCluster = pd.DataFrame({'genrecluster':[], 'pearson_revenue':[],'pearson_profit':[],'pearson_roi':[]})




folderName = 'genreClusterAnalysis'  
if os.path.exists(folderName):
    shutil.rmtree(folderName)    
os.makedirs(folderName)
for index,label in labelOnly.iterrows():
    #get label now 
   
    labelNow = label[0]
    print(labelNow)
    
    #select the data for label Now 
    dataLabelNow = dataframe[dataframe['label'] == labelNow] 
    
    #since the cluster genre already accurate ,get 1 of the data
    genreRep = dataLabelNow.genre[dataLabelNow.first_valid_index()]
    
    # create directory for this label for every purpose  
    thisLabelFolderName = 'genre-Label-' + str(labelNow) + '-'+ genreRep
    thisLabelDirPath = folderName+'\\'+thisLabelFolderName 
    if os.path.exists(thisLabelDirPath):
        shutil.rmtree(thisLabelDirPath)
    os.makedirs(thisLabelDirPath)
    
    
    #save the data 
    name_DataLabelNow = 'genre-'+'label-'+str(labelNow)+'_dataset.csv'
    dataLabelNow.to_csv(thisLabelDirPath+'\\'+name_DataLabelNow) 
    
    #generate WordCloud Actor from labelNow 
    split_column_actor = plotModule.split_column_actor(dataLabelNow)
    #trim actor cause there are 'kirsten dunst' and ' kirsten dunst' 
    split_column_actor['actors'] = split_column_actor['actors'].str.strip()
    
    actoronly = split_column_actor['actors']
    actoronly = actoronly.astype(str)
    np_actor = np.array(actoronly).astype(str)
    text_actor= " ".join(np_actor).lower() 
    
    wordCloudFileName = thisLabelDirPath + '\\wordcloud-genrelabel-'+str(labelNow)+'.jpg'
    plotModule.generateWordCloud(text_actor , wordCloudFileName ,'Actor Distribution for Genre Label cluster :' + str(labelNow))
    
    
    #boxPlot revenue for this cluster genre revenue estimation 
    boxplot_revenueName = thisLabelDirPath + '\\boxandwhiskerRevenue-genrelabel-'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['revenue'],False , 'Box and Whisker Revenue Distribution genre cluster-'+str(labelNow) , boxplot_revenueName) 
    
    #boxplot rating for this cluster rating estimation 
    boxplot_ratingName =  thisLabelDirPath + '\\boxandwhiskerRating-genrelabel-'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['rating'],False , 'Box and Whisker Rating Distribution genre cluster-'+str(labelNow) , boxplot_ratingName) 
    
    #boxplot budget for this cluster estimation 
    boxplot_budgetName  = thisLabelDirPath + '\\boxandwhiskerRoi-genrelabel-' + str(labelNow) +'.jpg'
    plotModule.boxplot(dataLabelNow['us_budget'] , False , 'Box and Whisker Budget Distribution genre cluster-'+ str(labelNow), boxplot_budgetName)
    
    
    #box plot profit for this cluster estimation 
    boxplot_profitName = thisLabelDirPath + '\\boxandwhiskerProfit-genrelabel-' + str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['profit'] , False , 'Box and Whisker Profit Distribution genre cluster-'+ str(labelNow) , boxplot_profitName) 
    
    #box plot roi for this cluster estimation 
    boxplot_roiName = thisLabelDirPath + '\\boxandwhiskerRoi-genrelabel-' + str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['roi'] , False , 'Box and Whisker Roi Distribution genre cluster-' + str(labelNow) , boxplot_roiName) 
    
    
    #boxplot youtubeView for this cluster estimation 
    boxplot_viewCountName = thisLabelDirPath + '\\boxandwhiskerviewCount-genrelabel-'+str(labelNow)+'.jpg'
    plotModule.boxplot(dataLabelNow['viewCount'], False , 'Box and Whisker view Count Youtube Distribution genre cluster-'+ str(labelNow), boxplot_viewCountName)
    
    
    #boxplot Instagram Hashtag Count for this cluster estimation 
    boxplot_InstagramHashtagCountName = thisLabelDirPath + '\\boxandwhiskerHashtagCountInstagram-genrelabel-'+str(labelNow)+'.jpg' 
    plotModule.boxplot(dataLabelNow['hashtagcount'] , False , 'Box and Whisker Instagram Count Hashtag Distribution genre cluster-'+ str(labelNow), boxplot_InstagramHashtagCountName) 
    
    
    # HYPOTHESIS GENRE 1. : Every genre combination have dominant actor 
    # count every actor play how many title 
    actor_title = split_column_actor[['actors' , 'title']]
    countAllActorInThisCluster = actor_title.groupby('actors').count() 
    # howManyFilmActorPlaysInCluster / howManyTitlesInTheCluster * 100
    countAllActorInThisCluster['contribution'] = countAllActorInThisCluster['title'] / dataLabelNow.shape[0] * 100 
    actorContributionDataFrameName = thisLabelDirPath + '\\cluster-'+str(labelNow)+'-genreTitleContribution.csv'
    #saveThisClusterActorContribution
    countAllActorInThisCluster.to_csv(actorContributionDataFrameName)
    #sort to get most contributing play in this cluster 
    countAllActorInThisCluster = countAllActorInThisCluster.sort_values('contribution', ascending=False) 
    mostPlayedActor = countAllActorInThisCluster.index[0]+'-'+str(labelNow)
    mostPlayedContribution = countAllActorInThisCluster.contribution[0] 
    
    #add to The Global DataFrame 
    actorContributionPerCluster = actorContributionPerCluster.append({"genrecluster":mostPlayedActor, "clustercontribution":mostPlayedContribution} , ignore_index=True)
    
    
    # HYPOTHESIS GENRE 2 : the dominant actor also contribute to the biggest in revenue 
    actor_revenue = split_column_actor[['actors', 'revenue']] 
    sumAllRevenuePerActorInThisCluster = actor_revenue.groupby('actors').sum() 
    # howManyFilmActorPlaysInCluster / howManyTitlesInTheCluster * 100
    sumAllRevenuePerActorInThisCluster['contribution'] = sumAllRevenuePerActorInThisCluster['revenue'] / dataLabelNow['revenue'].sum() * 100
    sumAllRevenuePerActorInThisCluster = sumAllRevenuePerActorInThisCluster.sort_values('contribution' , ascending = False)
    sumActorRevenueDataFrameName = thisLabelDirPath + '\\cluster-'+str(labelNow)+'-actorContributionRevenue.csv'
    sumAllRevenuePerActorInThisCluster.to_csv(sumActorRevenueDataFrameName)
    
    thisGenreCluster_favActor = genreRep + '-' + mostPlayedActor
    thisGenreCluster_favRevenueContribution = sumAllRevenuePerActorInThisCluster.contribution[countAllActorInThisCluster.index[0]]
    actorRevenueContributionPerCluster = actorRevenueContributionPerCluster.append({'genrecluster':thisGenreCluster_favActor, 'clustercontribution':thisGenreCluster_favRevenueContribution}, ignore_index = True)
    
    
    # HYPOTHESIS PROFIT IN CLUSTER GENRE : the dominant actor also contibute to the bigges
    actor_profit = split_column_actor[['actors', 'profit']] 
    sumAllProfitPerActorInThisCluster = actor_profit.groupby('actors').sum() 
    #howMuchProfitActorPlaysinThisCluster / HowMuchProfitInThisCluster * 100 
    sumAllProfitPerActorInThisCluster['contribution'] = sumAllProfitPerActorInThisCluster['profit'] / dataLabelNow['profit'].sum() * 100 
    sumAllProfitPerActorInThisCluster = sumAllProfitPerActorInThisCluster.sort_values('contribution' , ascending= False) 
    sumActorProfitDataFrameName = thisLabelDirPath + '\\cluster-' + str(labelNow)+'-actorContributionProfit.csv'
    sumAllProfitPerActorInThisCluster.to_csv(sumActorProfitDataFrameName)
    
    thisGenreCluster_favProfitContribution = sumAllProfitPerActorInThisCluster.contribution[countAllActorInThisCluster.index[0]]
    actorProfitContributionPerCluster = actorProfitContributionPerCluster.append({'genrecluster':thisGenreCluster_favActor, 'clustercontribution':thisGenreCluster_favProfitContribution}, ignore_index = True) 
    
    #HYPOTHESIS ROI IN CLUSTER GENRE : the dominant genre also contribute to the biggest 
    actor_roi = split_column_actor[['actors' , 'roi']] 
    sumAllRoiPerActorInThisCluster = actor_roi.groupby('actors').sum() 
    #howmuchroiactorPlayesInThisCluster / HowMuchRoiInThisCluster * 100 
    sumAllRoiPerActorInThisCluster['contribution'] = sumAllRoiPerActorInThisCluster['roi'] / dataLabelNow['roi'].sum() * 100
    sumAllRoiPerActorInThisCluster = sumAllRoiPerActorInThisCluster.sort_values('contribution' , ascending = False) 
    sumActorRoiDataFrameName = thisLabelDirPath + '\\cluster-' + str(labelNow) + '-actorContributionRoi.csv' 
    sumAllRoiPerActorInThisCluster.to_csv(sumActorRoiDataFrameName)
    
    thisGenreCluster_favRoiContribution = sumAllRoiPerActorInThisCluster.contribution[countAllActorInThisCluster.index[0]] 
    actorRoiContributionPerCluster = actorRoiContributionPerCluster.append({'genrecluster':thisGenreCluster_favActor, 'clustercontribution':thisGenreCluster_favRoiContribution} , ignore_index = True) 
    
    
     # BARCHART PEARSON YOUTUBEVIEW and InstagramHashtagCount 
    if dataLabelNow.shape[0] > 1:
        pearson_youtubecount_roi = pearsonr(dataLabelNow['roi'], dataLabelNow['viewCount'])[0]
        pearson_youtubecount_revenue = pearsonr(dataLabelNow['revenue'], dataLabelNow['viewCount'])[0] 
        pearson_youtubecount_profit = pearsonr(dataLabelNow['profit'] , dataLabelNow['viewCount'])[0]
        
        
        pearson_instagramcounthashtag_roi = pearsonr(dataLabelNow['roi'], dataLabelNow['hashtagcount'])[0]
        pearson_instagramcounthashtag_revenue = pearsonr(dataLabelNow['revenue'], dataLabelNow['hashtagcount'])[0]
        pearson_instagramcounthashtag_profit = pearsonr(dataLabelNow['profit'], dataLabelNow['hashtagcount'])[0]
        
        youtubeViewPearson_perCluster = youtubeViewPearson_perCluster.append({'genrecluster':genreRep+'-'+str(labelNow), 'pearson_revenue':pearson_youtubecount_revenue
                                                                              ,'pearson_profit':pearson_youtubecount_profit,'pearson_roi':pearson_youtubecount_roi}, ignore_index =True) 
    
        instagramHashtagPearson_perCluster =instagramHashtagPearson_perCluster.append({'genrecluster':genreRep+'-'+str(labelNow), 'pearson_revenue':pearson_instagramcounthashtag_revenue
                                                           ,'pearson_profit':pearson_instagramcounthashtag_profit,'pearson_roi':pearson_instagramcounthashtag_roi}, ignore_index = True)

    

# HYPOTHESIS GENRE 1 : every genre combination 
# saved all The most Played actor representatives 
actorContributionPerCluster = actorContributionPerCluster.sort_values('clustercontribution',ascending = False)
actorContributionPerCluster.to_csv('allClusterGenre-mostPlayedActor.csv')
#visualize 
plotModule.barcharth(actorContributionPerCluster['genrecluster'] , actorContributionPerCluster['clustercontribution'], 'genre cluster \n actor representatives' , 'contribution (Percent)' , 'Barchart Every Genre Cluster Most Played Actor', 'barchartMostPlayedActor.jpg')

biggerThan50 = actorContributionPerCluster[actorContributionPerCluster['clustercontribution'] >= 50] 
lowerThan50  = actorContributionPerCluster[actorContributionPerCluster['clustercontribution'] < 50] 

pieChartActorTitleContribution = pd.DataFrame({'label':[], 'count':[]}) 
pieChartActorTitleContribution = pieChartActorTitleContribution.append({'label':'clusterGenreWithActorThat \n PlayedMoreThan50Percent', 'count':biggerThan50.shape[0]}, ignore_index=True)
pieChartActorTitleContribution = pieChartActorTitleContribution.append({'label':'clusterGenreWithActorThat \n PlayedLessThan50Percent', 'count':lowerThan50.shape[0]}, ignore_index=True)
# plot the Disttribution for  comparison
plotModule.pieChart('Actor Representative Cluster Genre Comparison',pieChartActorTitleContribution['label'], pieChartActorTitleContribution['count'])


#HYPOTHESIS GENRE 2: actor reprsentatives also contribute to high revenue 
actorRevenueContributionPerCluster = actorRevenueContributionPerCluster.sort_values('clustercontribution', ascending = False) 
actorRevenueContributionPerCluster.to_csv('allClusterGenre-favActorRevenueContribution.csv')
plotModule.barcharth(actorRevenueContributionPerCluster['genrecluster'] , 
                     actorRevenueContributionPerCluster['clustercontribution'], 
                     'genre cluster  fav actor representatives \n revenue Contribution' ,
                     'contribution (Percent)' , 
                     'Barchart Every Genre Cluster Most Played Actor',
                     'barchartMostPlayedActor_revenueContribution.jpg')

revenueConBiggerThan50 = actorRevenueContributionPerCluster[actorRevenueContributionPerCluster['clustercontribution'] >=50]
revenueConLowerThan50  = actorRevenueContributionPerCluster[actorRevenueContributionPerCluster['clustercontribution'] < 50]


pieChartActor_SumRevenueContribution = pd.DataFrame({'label':[], 'count':[]}) 
pieChartActor_SumRevenueContribution = pieChartActor_SumRevenueContribution.append({'label':'favoriteActorPerGenreCluster \n Revenue contribute more than 50 percent', 'count':revenueConBiggerThan50.shape[0]} , ignore_index = True)
pieChartActor_SumRevenueContribution =  pieChartActor_SumRevenueContribution.append({'label':'favoriteActorPerGenreCluster \n Revenue contribute less than 50 percent', 'count':revenueConLowerThan50.shape[0]} , ignore_index = True)
plotModule.pieChart('Revenue Actor Representative Cluster Genre Comparison',pieChartActor_SumRevenueContribution['label'], pieChartActor_SumRevenueContribution['count'])

# HYPOTHESIS GENRE PROFIT : actor representatives also contribute to highest profit 
actorProfitContributionPerCluster = actorProfitContributionPerCluster.sort_values('clustercontribution', ascending = False) 
actorProfitContributionPerCluster.to_csv('allClusterGenre-favActorProfitContribution.csv') 
plotModule.barcharth(actorProfitContributionPerCluster['genrecluster'],
                    actorProfitContributionPerCluster['clustercontribution'],
                    'genre cluster fav actor representatives \n profit Contribution',
                    'contribution(Percent)' , 
                    'Barchart Every Genre Cluster Actor Profit Contribution',
                    'barchartMostPlayedActor_profitContribution.jpg') 
                    
profitConBiggerThan50 = actorProfitContributionPerCluster[actorProfitContributionPerCluster['clustercontribution']>=50] 
profitConLowerThan50  = actorProfitContributionPerCluster[actorProfitContributionPerCluster['clustercontribution'] < 50] 

pieChartActor_sumProfitContribution = pd.DataFrame({'label':[], 'count':[]})
pieChartActor_sumProfitContribution = pieChartActor_sumProfitContribution.append({'label':'favouriteActorPerGenreCluster \n Profit contribute more than 50 percent', 'count':profitConBiggerThan50.shape[0]}, ignore_index = True)
pieChartActor_sumProfitContribution = pieChartActor_sumProfitContribution.append({'label':'favouriteActorPerGenreCluster \n Profit contribute less than 50 percent', 'count':profitConLowerThan50.shape[0]}, ignore_index = True)
plotModule.pieChart('Profit Actor Representative Cluster Genre Comparison',pieChartActor_sumProfitContribution['label'] , pieChartActor_sumProfitContribution['count'])



#HYPOTHESIS GENRE ROI : actor Representatives also contribute to highest ROI 
actorRoiContributionPerCluster = actorRoiContributionPerCluster.sort_values('clustercontribution' , ascending = False) 
actorRoiContributionPerCluster.to_csv('allClusterGenre-favActorRoiContribution.csv') 
plotModule.barcharth(actorRoiContributionPerCluster['genrecluster'],
                     actorRoiContributionPerCluster['clustercontribution'],
                     'genre cluster fav actor representatives \n roi contribution',
                     'contribution (Percent)',
                     'Barchart every Genre Cluster Actor ROI Contribution', 
                     'barchartMostPlayedActor_roiContribution.jpg')

roiConBiggerThan50 = actorRoiContributionPerCluster[actorRoiContributionPerCluster['clustercontribution']>=50] 
roiConLowerThan50  = actorRoiContributionPerCluster[actorRoiContributionPerCluster['clustercontribution']< 50]

pieChartActor_sumRoiContribution = pd.DataFrame({'label':[], 'count':[]}) 
pieChartActor_sumRoiContribution = pieChartActor_sumRoiContribution.append({'label':'favouriteActorPerGenreCluster \n ROI contribute more than 50 percent', 'count':roiConBiggerThan50.shape[0]}, ignore_index = True)
pieChartActor_sumRoiContribution = pieChartActor_sumRoiContribution.append({'label':'favouriteActorPerGenreCluster \n ROI contribute less than 50 percent', 'count':roiConLowerThan50.shape[0]}, ignore_index = True)
plotModule.pieChart('Roi Actor Representatives Cluster Genre Comparison', pieChartActor_sumRoiContribution['label'], pieChartActor_sumRoiContribution['count'])





# youtube Trailer view to Other REsponse every cluster 
youtubeViewPearson_perCluster = youtubeViewPearson_perCluster.sort_values('pearson_revenue', ascending = False)
youtubeViewPearson_perCluster.to_csv('pearsonAllClusterYoutubeViewandAllResponseDataset.csv')
plotModule.barcharth(youtubeViewPearson_perCluster['genrecluster'],
                     youtubeViewPearson_perCluster['pearson_revenue'],
                     'genre Label', 'Pearson score',
                     'Pearson Score Barchart Every \n Cluster Youtube Views and Revenue', 'pearsonAllClusterYoutubeViewandRevenue.jpg')

instagramHashtagPearson_perCluster = instagramHashtagPearson_perCluster.sort_values('pearson_revenue', ascending = False)
instagramHashtagPearson_perCluster.to_csv('pearsonAllClusterInstagramHashtagCountandAllResponseDataset.csv')
plotModule.barcharth(instagramHashtagPearson_perCluster['genrecluster'],
                     instagramHashtagPearson_perCluster['pearson_revenue'], 
                     'genre Label', 'pearson Score',
                     'Pearson Score Barchart Every \n Cluster Instagram Hashtag Count and revenue', 'pearsonAllClusterInstagramHashtagCountandRevenue.jpg')


