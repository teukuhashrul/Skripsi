# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:49:17 2020

@author: lenovo
"""


import pandas as pd 
import matplotlib.pyplot as plt 
import plotModule
import numpy as np 
import os 
import shutil

dataset = pd.read_csv('hasilEksperimen_FinalForm.csv')


all_genre_combination = dataset.genre
all_genre_combination = all_genre_combination.drop_duplicates() 


# genre terpisah yang unik menggunakan pie chart 
split_genre_dataset = plotModule.split_column_genre(dataset)
# count groupby genre 
count_groupbygenre = split_genre_dataset[['genre', 'title']].groupby('genre').count() 
plotModule.pieChart('Genre Title Movies Distribution' , count_groupbygenre.index , count_groupbygenre.title) 


dataset_combination_count = pd.DataFrame({'genre':[], 'count':[]})
for genreNow in all_genre_combination:
    this_genre_data = dataset[dataset['genre'] == genreNow]
    count           = this_genre_data.shape[0] 
    
    dataset_combination_count  = dataset_combination_count.append({'genre':genreNow , 'count':count},ignore_index = True)


dataset_combination_count = dataset_combination_count.sort_values('count',ascending = False)

q2 = dataset_combination_count.quantile(0.5)
q1 = dataset_combination_count.quantile(0.25)
q3 = dataset_combination_count.quantile(0.75)

# How Many member in every genre combination
plt.title("Genre Combination Distribution Box and Whisker")
plt.boxplot(dataset_combination_count['count'] , vert=True) 
plt.show()
plt.clf()


# filter dataset
filtered_dataset = pd.DataFrame({}) 
for genreNow in dataset_combination_count.genre:
    countNow = dataset[dataset['genre'] == genreNow].shape[0] 
    if(countNow>2):
        dataGenreNow = dataset[dataset['genre'] == genreNow]
        filtered_dataset = filtered_dataset.append(dataGenreNow)


dataset = filtered_dataset
# top N revenue tiap kombinasi genre                    
all_genre_combination = dataset.genre 

genre_allRevenueBoxPlotDataFrame = pd.DataFrame({'genre':[],'revenueRange':[], 'revenueMean':[]})
all_genre_combination = all_genre_combination.drop_duplicates() 
for genreNow in all_genre_combination:
    this_genre_data = dataset[dataset['genre'] == genreNow]
    
    this_mean = this_genre_data.revenue.mean() 
    genre_allRevenueBoxPlotDataFrame = genre_allRevenueBoxPlotDataFrame.append({'genre':genreNow,
                                                                                 'revenueRange': this_genre_data.revenue,
                                                                                'revenueMean':this_mean}, ignore_index = True)
genre_allRevenueBoxPlotDataFrame = genre_allRevenueBoxPlotDataFrame.sort_values('revenueMean', ascending = False)


fig,ax = plt.subplots(figsize=(10,30)) 
plt.title('All Genre Revenue Combination \n')
plt.ylabel('Genre Label')
plt.xlabel('Revenue Distribution (Million Dollars)')
ax.set_yticklabels(genre_allRevenueBoxPlotDataFrame.genre)
ax.boxplot(genre_allRevenueBoxPlotDataFrame.revenueRange, vert=False) 

# top 10 Genre Revenue 
fig,ax = plt.subplots(figsize=(10,10)) 
plt.title('Top 10 All Genre Revenue Combination \n')
plt.ylabel('Genre Label')
plt.xlabel('Revenue Distribution (Million Dollars)')
ax.set_yticklabels(genre_allRevenueBoxPlotDataFrame.head(10).genre)
ax.boxplot(genre_allRevenueBoxPlotDataFrame.head(10).revenueRange, vert=False) 


## Top N jumlah genre yang paling banyak dibuat terpisah
countTitle_Genre = dataset[['genre' , 'title']].groupby('genre').count() 
countTitle_Genre = countTitle_Genre.sort_values('title', ascending = False)



## Semua multi barchart 
## ada dataframe rata rata revenue , rata rata budget , sama selisih rata rata nya 
## bikin yang gede satu sama yang top 10 satu 
all_genre_combination = dataset.genre 
all_genre_combination = all_genre_combination.drop_duplicates() 
profitableDataFrame = pd.DataFrame({'genre':[], 'meanRevenue':[], 'meanBudget':[], 'difference':[]})
for genreNow in all_genre_combination: 
    this_Genre_Data = dataset[dataset['genre'] == genreNow] 
    
    thisMeanRevenue = np.mean(this_Genre_Data.revenue)
    thisMeanBudget  = np.mean(this_Genre_Data.us_budget) 
    
    difference = thisMeanRevenue - thisMeanBudget
    profitableDataFrame = profitableDataFrame.append({'genre':genreNow, 'meanRevenue':thisMeanRevenue, 'meanBudget':thisMeanBudget
                                                      , 'difference':difference}, ignore_index = True)

profitableDataFrame = profitableDataFrame.sort_values('difference' , ascending = False) 

topN_profitableDataFrame = profitableDataFrame.head(10)
barLabels = ['revenue', 'budget'] 
Xlabels   = topN_profitableDataFrame.genre 
MultiYData = [topN_profitableDataFrame.meanRevenue ,topN_profitableDataFrame.meanBudget]

plotModule.multiBarchart(Xlabels , MultiYData, barLabels,'Genre', 'Million Dollars'
                         , 'Top 10 Genre Movie with High profit')
# =============================================================================
# 
# =============================================================================
## Top 10 all Genre with high Votes 
all_genre_combination = dataset.genre 
all_genre_combination = all_genre_combination.drop_duplicates() 
rankedVotesGenreDataFrame = pd.DataFrame({'genre':[], 'votesRange':[], 'meanVotes':[]})
for genreNow in all_genre_combination:
    this_genreData = dataset[dataset['genre'] == genreNow] 
    meanVotes = np.mean(this_genreData.votes) 
    rankedVotesGenreDataFrame = rankedVotesGenreDataFrame.append({'genre':genreNow, 'votesRange':this_genreData.votes, 'meanVotes':meanVotes},ignore_index = True)

    
rankedVotesGenreDataFrame=rankedVotesGenreDataFrame.sort_values('meanVotes', ascending = False)

fig,ax = plt.subplots(figsize=(10,30)) 
plt.title('All Genre Votes Combination \n')
plt.ylabel('Genre Label')
plt.xlabel('Votes Distribution')
ax.set_yticklabels(rankedVotesGenreDataFrame.genre)
ax.boxplot(rankedVotesGenreDataFrame.votesRange, vert=False) 

plt.clf()
# top 10 Genre Revenue 
fig,ax = plt.subplots(figsize=(10,10)) 
plt.title('Top 10 All Genre Votes Combination \n')
plt.ylabel('Genre Label')
plt.xlabel('Votes Distribution')
ax.set_yticklabels(rankedVotesGenreDataFrame.head(10).genre)
ax.boxplot(rankedVotesGenreDataFrame.head(10).votesRange, vert=False)



# US_BUDGET FOR ALL MOVIE TREND 
sumBudget_GroupYear = dataset[['year', 'us_budget']].groupby('year').sum() 

plt.bar(sumBudget_GroupYear.index , sumBudget_GroupYear['us_budget'],edgecolor='black')
plt.xlabel('Year',fontweight='bold')
plt.ylabel('Us_Budget',fontweight='bold')
plt.title('Budget Invested Year by Year' , fontweight='bold')
#set axis if needed
#plt.ylim(0,1)
plt.xticks(rotation=90)
    
#draw line 
#plt.axhline(y=0.0, color='r' , linestyle='-')
plt.show()
plt.clf()



## Barchart Top Genre Film tiap tahun 
## kombinasi genre paling banyak dibuat dari tahun ke tahun 

## kombinasi genre dengan 
all_year = dataset.year 
all_year = all_year.drop_duplicates()
mostTitleGenreCombinationPerYearDataFrame = pd.DataFrame({'year':[], 'yeargenre':[], 'count':[]})
for yearNow in all_year: 
    this_yearData  = dataset[dataset['year'] == yearNow]
    countTitleByGenre = this_yearData[['genre','title']].groupby('genre').count()
    countTitleByGenre = countTitleByGenre.sort_values('title',ascending= False)
    
    topGenreThisYear = countTitleByGenre.index[0] 
    topGenreCount    = countTitleByGenre.title[0] 
    mostTitleGenreCombinationPerYearDataFrame = mostTitleGenreCombinationPerYearDataFrame.append({'year':yearNow
    ,'yeargenre':topGenreThisYear+'\n'+str(yearNow),'count':topGenreCount}, ignore_index = True)
    
mostTitleGenreCombinationPerYearDataFrame = mostTitleGenreCombinationPerYearDataFrame.sort_values('year')


plt.figure(figsize=(27,8))
plt.bar(mostTitleGenreCombinationPerYearDataFrame['yeargenre'] , mostTitleGenreCombinationPerYearDataFrame['count'],edgecolor='black')
plt.xlabel('Year and Genre',fontweight='bold')
plt.ylabel('How Many Movie Created',fontweight='bold')
plt.title('Most Movie Created year by year' , fontweight='bold')
plt.show()
plt.clf()


## kombinasi genre dengan rata rata revenue dari tahun ke tahun 
highestRevenueMeanGenreCombinationPerYearDataFrame = pd.DataFrame({'year':[], 'yeargenre':[], 'meanRevenue':[]}) 
for yearNow in all_year: 
    this_yearData = dataset[dataset['year'] == yearNow]
    meanRevenueByGenre = this_yearData[['genre', 'revenue']].groupby('genre').mean() 
    meanRevenueByGenre = meanRevenueByGenre.sort_values('revenue', ascending = False) 
    
    topGenreThisYear = meanRevenueByGenre.index[0] 
    topGenreMeanRevenue = meanRevenueByGenre.revenue[0] 
    
    highestRevenueMeanGenreCombinationPerYearDataFrame = highestRevenueMeanGenreCombinationPerYearDataFrame.append({'year':yearNow, 'yeargenre':topGenreThisYear+'\n'+str(yearNow)
                                                                                                                    , 'meanRevenue':topGenreMeanRevenue}, ignore_index= True)
    
highestRevenueMeanGenreCombinationPerYearDataFrame = highestRevenueMeanGenreCombinationPerYearDataFrame.sort_values('year') 

plt.figure(figsize=(29,8))
plt.bar(highestRevenueMeanGenreCombinationPerYearDataFrame['yeargenre'] ,highestRevenueMeanGenreCombinationPerYearDataFrame['meanRevenue'],edgecolor='black')
plt.xlabel('Year and Genre',fontweight='bold')
plt.ylabel('Mean Revenue',fontweight='bold')
plt.title('Most High Mean Revenue Genre Combination Year by Year' , fontweight='bold')
plt.show()
plt.clf() 

# =============================================================================
# 
# =============================================================================

## kombinasi genre dengan ratarata profit tertinggi dari tahun ke tahun 
highestProfitMeanGenreCombinationPerYearDataFrame = pd.DataFrame({'year':[], 'yeargenre':[], 'meanProfit':[]})
for yearNow in all_year:
    this_yearData = dataset[dataset['year']== yearNow] 
    meanProfitByGenre = this_yearData[['genre', 'profit']].groupby('genre').mean() 
    meanProfitByGenre = meanProfitByGenre.sort_values('profit', ascending = False) 
    
    topGenreThisYear= meanProfitByGenre.index[0]
    topGenreMeanProfit = meanProfitByGenre.profit[0] 
    
    highestProfitMeanGenreCombinationPerYearDataFrame = highestProfitMeanGenreCombinationPerYearDataFrame.append({'year':yearNow,
                                                                                                                  'yeargenre':topGenreThisYear+'\n'+str(yearNow), 'meanProfit':topGenreMeanProfit},ignore_index = True)

highestProfitMeanGenreCombinationPerYearDataFrame = highestProfitMeanGenreCombinationPerYearDataFrame.sort_values('year') 
   
plt.figure(figsize=(29,8))
plt.bar(highestProfitMeanGenreCombinationPerYearDataFrame['yeargenre'] ,highestProfitMeanGenreCombinationPerYearDataFrame['meanProfit'],edgecolor='black')
plt.xlabel('Year and Genre',fontweight='bold')
plt.ylabel('Mean Profit',fontweight='bold')
plt.title('Most High Mean Profit Genre Combination Year by Year' , fontweight='bold')
plt.show()
plt.clf() 


# =============================================================================
#  ## kombinasi genre dengan ratarata roi tertinggi dari tahun ke tahun 
# =============================================================================
highestRoiMeanGenreCombinationPerYearDataFrame = pd.DataFrame({'year':[], 'yeargenre':[], 'meanRoi':[]}) 
for yearNow in all_year:
    this_yearData = dataset[dataset['year']==yearNow] 
    meanRoiByGenre = this_yearData[['genre', 'roi']].groupby('genre').mean() 
    meanRoiByGenre = meanRoiByGenre.sort_values('roi', ascending = False) 
    
    topGenreThisYear = meanRoiByGenre.index[0] 
    topGenreMeanRoi = meanRoiByGenre.roi[0] 
    
    
    highestRoiMeanGenreCombinationPerYearDataFrame =  highestRoiMeanGenreCombinationPerYearDataFrame.append({'year':yearNow, 'yeargenre':topGenreThisYear+'\n'+str(yearNow),
                                                                                                    'meanRoi':topGenreMeanRoi}, ignore_index = True) 
    
    
highestRoiMeanGenreCombinationPerYearDataFrame = highestRoiMeanGenreCombinationPerYearDataFrame.sort_values('year')

plt.figure(figsize=(29,8))
plt.bar(highestRoiMeanGenreCombinationPerYearDataFrame['yeargenre'] ,highestRoiMeanGenreCombinationPerYearDataFrame['meanRoi'],edgecolor='black')
plt.xlabel('Year and Genre',fontweight='bold')
plt.ylabel('Mean Roi (%)',fontweight='bold')
plt.title('Most High Mean Roi Genre Combination Year by Year' , fontweight='bold')
plt.show()
plt.clf() 

# =============================================================================
# TOP 3 EVERY YEAR QUANTILE REVENUE
# =============================================================================
# top 3 revenue year by quantile
top_3CombinationGenreDataFrame = pd.DataFrame({'year':[],'genre':[], 'quantileRevenue':[]})
for yearNow in all_year: 
    this_yearData = dataset[dataset['year']== yearNow] 
    quantileRevenueByGenre = this_yearData[['genre','revenue']].groupby('genre').quantile()
    quantileRevenueByGenre  = quantileRevenueByGenre.sort_values('revenue', ascending = False) 
    
    top1_genre = quantileRevenueByGenre.index[0]
    top1_revenue = quantileRevenueByGenre.revenue[0] 
    
    top2_genre = quantileRevenueByGenre.index[1] 
    top2_revenue = quantileRevenueByGenre.revenue[1] 
    
    top3_genre = quantileRevenueByGenre.index[2]
    top3_revenue = quantileRevenueByGenre.revenue[2]
    
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top1_genre
                                                                           ,'quantileRevenue':top1_revenue},ignore_index = True)
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top2_genre
                                                                           ,'quantileRevenue':top2_revenue},ignore_index = True)
    
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top3_genre
                                                                           ,'quantileRevenue':top3_revenue},ignore_index = True)


fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(15)
plt.xlabel('year')
plt.ylabel('quantileRevenue')
plt.title("Top 3 Trend Top Genre With Most Revenue ")
distinct_genre = top_3CombinationGenreDataFrame.genre.drop_duplicates()

markers = ["." , ",", "o" , "v", "^", "<", ">", "1"  , "2"
	   , "3", "4", "8" ,"s" , "p", "P" , "*", "h" , "H", 
	  "+", "x", "X" , "D", "d"]

marker_i = 0
for genreNow in distinct_genre:
    n= []
    dataNow = top_3CombinationGenreDataFrame[top_3CombinationGenreDataFrame['genre']==genreNow] 
    x = np.array(dataNow['year'])
    y = np.array(dataNow['quantileRevenue'])
    ax.scatter(x,y, label=genreNow , marker = markers[marker_i], s=150)
    marker_i = marker_i + 1

    for genre in dataNow.genre:
        arr_genre = genre.split(',')
        new_genre = ''
        for text in arr_genre:
            new_genre = new_genre + ' ' + text[0:2] + ','
    
        n.append(new_genre)
      
ax.legend(loc='upper center' , ncol=4, fancybox = True)

# =============================================================================
#  TOP 3 EVERY YEAR QUANTILE PROFIT
# =============================================================================

# top 3 profit year by quantile
top_3CombinationGenreDataFrame = pd.DataFrame({'year':[],'genre':[], 'quantileProfit':[]})
for yearNow in all_year: 
    this_yearData = dataset[dataset['year']== yearNow] 
    quantileProfitByGenre = this_yearData[['genre','profit']].groupby('genre').quantile()
    quantileProfitByGenre  = quantileProfitByGenre.sort_values('profit', ascending = False) 
    
    top1_genre = quantileProfitByGenre.index[0]
    top1_profit = quantileProfitByGenre.profit[0] 
    
    top2_genre = quantileProfitByGenre.index[1] 
    top2_profit = quantileProfitByGenre.profit[1] 
    
    top3_genre = quantileProfitByGenre.index[2]
    top3_profit = quantileProfitByGenre.profit[2]
    
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top1_genre
                                                                           ,'quantileProfit':top1_profit},ignore_index = True)
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top2_genre
                                                                           ,'quantileProfit':top2_profit},ignore_index = True)
    
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top3_genre
                                                                           ,'quantileProfit':top3_profit},ignore_index = True)


fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(15)
plt.xlabel('year')
plt.ylabel('quantileProfit')
plt.title("Top 3 Trend Top Genre With Most Profit ")
distinct_genre = top_3CombinationGenreDataFrame.genre.drop_duplicates()

markers = ["." , ",", "o" , "v", "^", "<", ">", "1"  , "2"
	   , "3", "4", "8" ,"s" , "p", "P" , "*", "h" , "H", 
	  "+", "x", "X" , "D", "d","." , ",", "o" , "v", "^", "<",]

marker_i = 0
for genreNow in distinct_genre:
    n= []
    dataNow = top_3CombinationGenreDataFrame[top_3CombinationGenreDataFrame['genre']==genreNow] 
    x = np.array(dataNow['year'])
    y = np.array(dataNow['quantileProfit'])
    ax.scatter(x,y, label=genreNow , marker = markers[marker_i], s=150, zorder=2)
    marker_i = marker_i + 1

    for genre in dataNow.genre:
        arr_genre = genre.split(',')
        new_genre = ''
        for text in arr_genre:
            new_genre = new_genre + ' ' + text[0:2] + ','
    
        n.append(new_genre)
        

ax.legend(loc='upper center' , ncol=4, fancybox = True)


# =============================================================================
#  TOP 3 EVERY YEAR QUANTILE ROI
# =============================================================================

top_3CombinationGenreDataFrame = pd.DataFrame({'year':[],'genre':[], 'quantileRoi':[]})
for yearNow in all_year: 
    this_yearData = dataset[dataset['year']== yearNow] 
    quantileRoiByGenre = this_yearData[['genre','roi']].groupby('genre').quantile()
    quantileRoiByGenre  = quantileRoiByGenre.sort_values('roi', ascending = False) 
    
    top1_genre = quantileRoiByGenre.index[0]
    top1_roi = quantileRoiByGenre.roi[0] 
    
    top2_genre = quantileRoiByGenre.index[1] 
    top2_roi = quantileRoiByGenre.roi[1] 
    
    top3_genre = quantileRoiByGenre.index[2]
    top3_roi = quantileRoiByGenre.roi[2]
    
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top1_genre
                                                                           ,'quantileRoi':top1_roi},ignore_index = True)
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top2_genre
                                                                           ,'quantileRoi':top2_roi},ignore_index = True)
    
    top_3CombinationGenreDataFrame =top_3CombinationGenreDataFrame.append({'year':yearNow
                                                                           ,'genre':top3_genre
                                                                           ,'quantileRoi':top3_roi},ignore_index = True)


fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(15)
plt.xlabel('year')
plt.ylabel('quantileProfit')
plt.title("Top 3 Trend Top Genre With Most Roi ")
distinct_genre = top_3CombinationGenreDataFrame.genre.drop_duplicates()

markers = ["." , ",", "o" , "v", "^", "<", ">", "1"  , "2"
	   , "3", "4", "8" ,"s" , "p", "P" , "*", "h" , "H", 
	  "+", "x", "X" , "D", "d"]

marker_i = 0
for genreNow in distinct_genre:
    n= []
    dataNow = top_3CombinationGenreDataFrame[top_3CombinationGenreDataFrame['genre']==genreNow] 
    x = np.array(dataNow['year'])
    y = np.array(dataNow['quantileRoi'])
    ax.scatter(x,y, label=genreNow , marker = markers[marker_i], s=250)
    marker_i = marker_i + 1

    for genre in dataNow.genre:
        arr_genre = genre.split(',')
        new_genre = ''
        for text in arr_genre:
            new_genre = new_genre + ' ' + text[0:2] + ','
    
        n.append(new_genre)


top_3CombinationGenreDataFrame = top_3CombinationGenreDataFrame.sort_values(['genre' , 'year']) 
arr = top_3CombinationGenreDataFrame.index
color = ['b','g', 'r', 'cyan', 'm', 'y', 'b']
coloridx = 0
i = 0 
while(i < len(arr)-1):
    a_genre = top_3CombinationGenreDataFrame.genre[arr[i]]
    b_genre = top_3CombinationGenreDataFrame.genre[arr[i+1]] 
    
    a_x = top_3CombinationGenreDataFrame.year[arr[i]] 
    a_y = top_3CombinationGenreDataFrame.quantileRoi[arr[i]]
        
    b_x =top_3CombinationGenreDataFrame.year[arr[i+1]]
    b_y = top_3CombinationGenreDataFrame.quantileRoi[arr[i+1]] 
    
    if(a_genre == b_genre and abs(a_x-b_x)==1): 
        ax.plot([a_x,b_x]  , [a_y,b_y], color =color[coloridx], zorder=0, linewidth = 3)
    else:
        coloridx = coloridx + 1
        if(coloridx > len(color)-1):
            coloridx = 0
        
        
    i = i+1
        
ax.legend(loc='upper center' , ncol=4, fancybox = True)


    
    
# GenerateAllCombinationGenre Trend Per Year :
folder= 'allGenreTrend'
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

for genreNow in all_genre_combination: 
    thisGenreDir = folder + '\\' + genreNow
    if os.path.exists(thisGenreDir):
        shutil.rmtree(thisGenreDir)
    os.makedirs(thisGenreDir)
    thisGenreDataNow = dataset[dataset['genre'] == genreNow] 
    thisGenreDataNow = thisGenreDataNow.sort_values('year')
    # get all Year 
    allYear = thisGenreDataNow.year 
    allYear = allYear.drop_duplicates()
    
    
    thisGenreSumAllFeatureDataFrame = pd.DataFrame({'year':[], 'sumRevenue':[], 'sumRoi':[], 'sumProfit':[], 'sumBudget':[] })
    for year in allYear:
        thisYearDataNow = thisGenreDataNow[thisGenreDataNow['year'] == year] 
        
        sumBudget = thisYearDataNow.us_budget.quantile()
        sumRevenue = thisYearDataNow.revenue.quantile() 
        sumProfit = thisYearDataNow.profit.quantile() 
        sumRoi = thisYearDataNow.roi.sum() 
        
        thisGenreSumAllFeatureDataFrame= thisGenreSumAllFeatureDataFrame.append({'year':year
                                                       , 'sumRevenue':sumRevenue
                                                       , 'sumRoi':sumRoi
                                                       , 'sumProfit':sumProfit
                                                       , 'sumBudget':sumBudget }, ignore_index = True)
    
    thisGenreFileDir = thisGenreDir + '\\' + 'yearTrendAnalysis.jpg' 
    thisGenreDataFrameDir = thisGenreDir + '\\' + genreNow+'DataFrame.csv'
    multiYLabel = ['Revenue' ,'Profit'] 
    multiYData  = [
                 thisGenreSumAllFeatureDataFrame['sumRevenue']
                 ,thisGenreSumAllFeatureDataFrame['sumProfit']]
    plotModule.saveMultiBarchart(thisGenreSumAllFeatureDataFrame['year'],
                             multiYData , multiYLabel , 'Year', 'Sum', genreNow+' YearTrendAnalysis', thisGenreFileDir)
   
    
    thisGenreSumAllFeatureDataFrame.to_csv(thisGenreDataFrameDir)
        
        
        
        
    