# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:02:03 2020

@author: lenovo
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shutil
import math
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import AgglomerativeClustering, KMeans
import os
from sklearn.metrics.pairwise import cosine_similarity , cosine_distances
import sys
np.set_printoptions(threshold=sys.maxsize)

def scatterplot(xVar , yVar, xName , yName , dirFileName): 
    plt.figure(figsize=(10,13  ))
    plt.scatter(xVar , yVar)
    plt.ylabel(yName)
    plt.xlabel(xName)
    plt.title("Scatter plot perbandingan {} dan {}".format(xName , yName), fontweight = "bold")
    # only choose 1 , show() to the console , savefig save to the folder 
  
    plt.savefig(dirFileName)
    plt.clf()


# method to create histogram on every numerical  feature 
def histogram(xData , xName , yName , num_bins , title , intervalX , dirToSave):   
    
    import numpy as np
    plt.figure(figsize=(10,8))
    plt.grid(zorder = 0) 
    
    # set interval for x 
    if intervalX is not None:
        plt.xticks(np.arange(min(xData) , max(xData) , intervalX))
    
    
    n , bins , patches = plt.hist(xData , num_bins ,facecolor = 'green' , alpha =0.5 ,linewidth = 1.2,
                                  edgecolor='black'  , zorder=3)                                             
    plt.xlabel(xName)
    plt.ylabel(yName) 
    # setting range histogram x axis plt.xticks(np.arange(lowest,highest , interval))
    #plt.xticks(np.arange(-10,850,100))
    plt.title(title ,fontweight = "bold")
    plt.savefig(dirToSave)
    #plt.show()
    plt.clf()
 

def barchart(xVar , yVar , xName , yName , title , dirName): 
    plt.figure(figsize=(10,13))
    plt.bar(xVar , yVar,edgecolor='black')
    plt.xlabel(xName,fontweight='bold')
    plt.ylabel(yName,fontweight='bold')
    plt.title(title , fontweight='bold')
    #set axis if needed
    #plt.ylim(0,1)
    plt.xticks(rotation=90)

    plt.savefig(dirName) 
    
    #draw line 
    #plt.axhline(y=0.0, color='r' , linestyle='-')
    #plt.show()
    plt.clf()
    
    
def boxplot(xData , isVertical , title , directoryPathToSave):
    plt.title(title)
    plt.boxplot(xData , vert=isVertical) 
    plt.savefig(directoryPathToSave)
    #plt.show()
    plt.clf()

 
def multiBarchart(xData, multiYData, multiYLabel , xLabel , yLabel , title):    
    plt.figure(figsize=(12,5)) 
    x = np.arange(len(xData))
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(rotation=90)
    plt.title(title)
    ax = plt.subplot(111)
    ax.set_xticks(x)
    ax.set_xticklabels(xData)


    interval = 0.2
    mid = math.ceil(len(multiYData)/2)
  
    
    nextCounter = 1
    for i in range(0 , len(multiYData)):    
        if(i<=mid-1):
            position = (i+1)*interval*-1
            ax.bar(x-0.2, multiYData[i], width=0.2, color='b', align='center',label=multiYLabel[i])
        elif(i>=mid-1):
            position = (nextCounter) * interval
            nextCounter = nextCounter + 1
            ax.bar(x+position, multiYData[i] , width = 0.2 , color = 'g', align ='center', label=multiYLabel[i])
        else:
            ax.bar(x,multiYData[i] , width = 0.2 , color = 'blue', align = 'center',label=multiYLabel[i])
    ax.legend()
    plt.show()
    plt.clf()

def saveMultiBarchart(xData, multiYData, multiYLabel , xLabel , yLabel , title, saveDir):    
    plt.figure(figsize=(15,10)) 
    x = np.arange(len(xData))
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(rotation=90)
    plt.title(title)
    ax = plt.subplot(111)
    ax.set_xticks(x)
    ax.set_xticklabels(xData)


    interval = 0.2
    mid = math.ceil(len(multiYData)/2)
  
    color = ['red', 'green', 'blue', 'yellow', 'black']
    nextCounter = 1
    for i in range(0 , len(multiYData)):    
        if(i<=mid-1):
            position = (i+1)*interval*-1
            ax.bar(x-0.2, multiYData[i], width=0.2, color=color[i], align='center',label=multiYLabel[i])
        elif(i>=mid-1):
            position = (nextCounter) * interval
            nextCounter = nextCounter + 1
            ax.bar(x+position, multiYData[i] , width = 0.2 , color =color[i], align ='center', label=multiYLabel[i])
        else:
            ax.bar(x,multiYData[i] , width = 0.2 , color = color[i], align = 'center',label=multiYLabel[i])
    ax.legend()
    plt.savefig(saveDir)
    plt.clf()


def pieChart(title,labels, values):
    plt.figure(figsize=(6,6))
    plt.title(title , fontweight='bold')
    x = labels
    y = values
    colors =['yellow' ,'darkcyan' ,'lightcoral' , 'darkorange' , 'darkgreen' ,'royalblue'  , 'sandybrown'
           ,'peru' , 'tomato' , 'tan' ,'hotpink', 'crimson' ,'mediumslateblue' ,'greenyellow'
           ,'firebrick' ,'darkcyan', 'orchid', 'azure' , 'chocolate' , 'slategrey']
    porcent = 100.*y/y.sum()
    patches, texts = plt.pie(y, colors = colors ,startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} % - {2} '.format(i,j, k) for i,j , k in zip(x, porcent,values)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))

    plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
# =============================================================================
# 
#     plt.savefig('piechart.png', bbox_inches='tight')            
#     plt.figure(figsize=(10,10))
#     plt.title(title , fontweight = "bold")    
#     plt.pie(values  , labels  = labels , autopct ='%1.1f%%' , shadow=True , startangle = 90)
#     plt.legend()
#     
# =============================================================================
    plt.show()
    plt.clf()
    
def barcharth(yLabel , xValue , yName , xName , title, directoryfilepath):
    # plt.figure(figsize=(15,30)) 
    plt.figure(figsize=(17,10)) 
    #
    
    plt.title(title,fontweight='bold')
    plt.barh(yLabel , xValue ,edgecolor='black')
    plt.xlabel(xName)
    plt.ylabel(yName)
    #plt.show()
    plt.savefig(directoryfilepath, dpi=150)
    plt.clf()
    

  
# split genre multiple row by comma delimiter 
def split_column_actor(inputdataset):  
    reshaped = (inputdataset.set_index(inputdataset.columns.drop('actors',1).tolist())
        .actors.str.split(',', expand=True)
        .stack()
        .reset_index()
        .rename(columns={0:'actors'})
        .loc[:, inputdataset.columns]
        )
    return reshaped

# split genre multiple row by comma delimiter 
def split_column_genre(inputdataset):  
    reshaped = (inputdataset.set_index(inputdataset.columns.drop('genre',1).tolist())
        .genre.str.split(',', expand=True)
        .stack()
        .reset_index()
        .rename(columns={0:'genre'})
        .loc[:, inputdataset.columns]
        )
    return reshaped
#split_genre = split_column_genre(dataset_)
#genreonly = split_genre['genre']
#genreonly = genreonly.astype(str)
#np_genre = np.array(genreonly).astype(str)
#text = " ".join(np_genre).lower()

#  function create wordcloud , input single text 
def generateWordCloud(text , pathToSave, title):
    wordcloud = WordCloud(repeat=False,collocations=False , background_color='white').generate(text)
    #display the generate image
    plt.title(title)
    plt.imshow(wordcloud , interpolation='bilinear')
    plt.axis('off')
    plt.savefig(pathToSave)
    #plt.show()
    plt.clf()
    
    
    
#call wordcloud 
#generateWordCloud(text , 'genre')  
    
# =============================================================================
#  
# =============================================================================
def multiBoxPlot(title , isVertical, yData ,yLabels , xLabels,xData):   
  
    fig,ax = plt.subplots() 
    plt.xlim(-1,1)
    plt.title(title)
    plt.ylabel(yLabels)
    plt.xlabel(xLabels)
    ax.set_yticklabels(yData)
    
    ax.boxplot(xData , vert=isVertical) 
    plt.show()
    plt.clf()
    
def saveMultiBoxPlot(title , isVertical, yData ,yLabels , xLabels,xData, dirToSave ):   
  
    fig,ax = plt.subplots() 
    plt.xlim(-1,1)
    plt.title(title)
    plt.ylabel(yLabels)
    plt.xlabel(xLabels)
    ax.set_yticklabels(yData)
    
    ax.boxplot(xData , vert=isVertical) 
    plt.savefig(dirToSave)
    plt.clf()

