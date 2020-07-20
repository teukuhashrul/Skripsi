# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 05:22:31 2020

@author: lenovo
"""

# =============================================================================
# this experiment is using to scrap data from IMDB 
# to complete data like budget 
# https://imdbpy.readthedocs.io/en/latest/
# =============================================================================
import pandas as pd
#if new machine using the imdb library dont forget to pip install imdbpy
import imdb

#set up imdb py object 
ia = imdb.IMDb() 

# read dataset 
dataset = pd.read_csv('hasileksperimen2.csv')

# create container for 
titleAndGross = pd.DataFrame({'title':[], 'budget':[]})
for title in dataset.title:
    print(title)
    # search keyword movie 
    movies = ia.search_movie(title) 
    #get one movie
    movieNow = ia.get_movie(movies[0].movieID)
    #know all the possible keys , because one movie is a dictionary with key and value 
    #guardian.infoset2keys
    #guardian.keys()
    budget = 0
    
    if movieNow.get('box office') is not None:
    
        #get box office data liek budget and gross from object movie 
        boxOfficeData = movieNow['box office'] 
        #box office consist of 3 item gross and budget using key 
       
        if boxOfficeData.get('Budget') is not None:
            budget = boxOfficeData['Budget']
    
    
    print('title : '+ title + ' budget : '+ str(budget))
    titleAndGross = titleAndGross.append({'title':title , 'budget':budget} , ignore_index = True)
    
    
movies = ia.search_movie('Into The Woods') 
    #get one movie
movieNow = ia.get_movie(movies[0].movieID)
    #know all the possible keys , because one movie is a dictionary with key and value 
    #guardian.infoset2keys
    #guardian.keys()
movieNow['title']




