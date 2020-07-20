# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:03:05 2020

@author: lenovo
"""
# =============================================================================
# Source Tutorial 
# activate API KEYS FIRST IN THE GOOGLE MANAGER
# https://www.dataquest.io/blog/python-api-tutorial/
# https://developers.google.com/youtube/v3/docs/search/list?apix_params=%7B%22part%22%3A%22snippet%22%2C%22q%22%3A%22the%20dark%20knight%20trailer%22%7D&apix=true
# =============================================================================
import requests
import json
import pandas as pd 

apikey = "AIzaSyBAHdAIu1-_l10NkgMEt_E8xbFDTcelRTw"
test_id  = "3J6o7hcm8bE"
response = requests.get('https://www.googleapis.com/youtube/v3/videos?part=statistics&id='+ test_id +'&key='+apikey)


# method to print in json object , so it will easier to read 
def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

# =============================================================================
# {'viewCount': '144043132',
#'likeCount': '1887384',
# 'dislikeCount': '92479',
# 'favoriteCount': '0',
# 'commentCount': '90586'}
# =============================================================================
response.json()['items'][0]['statistics']


#

query = 'the dark knight trailer'
search_response =  requests.get('https://www.googleapis.com/youtube/v3/search?part=snippet&q='+query+'&key='+apikey)

search_response.json()


jprint(search_response.json())

search_response.json()['items'][0]['id']['videoId']
title = search_response.json()['items'][0]['snippet']['title']

video_id     = search_response.json()['items'][0]['id']['videoId']
# get all the data statsitics 
    
statistic_response =  requests.get('https://www.googleapis.com/youtube/v3/videos?part=statistics&id='+ video_id +'&key='+apikey)

statistic_response.json()

# read data frame 
# seach get 
# get statistics from id 
# append to dataframe 

dataset = pd.read_csv('hasileksperiment-withROI.csv')
resultDataFrame = pd.DataFrame({'title':[],'youtubeTitle':[],
                                'viewCount_Youtube':[], 'likeCount_Youtube':[],
                                'dislikeCount_Youtube':[], 'commentCount_Youtube':[]})
for title in dataset.title:
   
    # search video 
    query = title + ' trailer' 
    print('searching : '+query)
    #search get id 
  
    search_response =  requests.get('https://www.googleapis.com/youtube/v3/search?part=snippet&q='+query+'&maxResults=1&key='+apikey) 
    youtubeTitle = search_response.json()['items'][0]['snippet']['title']
    video_id     = search_response.json()['items'][0]['id']['videoId']
    # get all the data statsitics 
    
    statistic_response =  requests.get('https://www.googleapis.com/youtube/v3/videos?part=statistics&id='+ video_id +'&key='+apikey)
    
    
    viewCount = statistic_response.json()['items'][0]['statistics']['viewCount']
    likeCount = 0 
    dislikeCount = 0 
    commentCout = 0
    if 'likeCount' in statistic_response.json()['items'][0]['statistics']:
        likeCount = statistic_response.json()['items'][0]['statistics']['likeCount']
    if 'dislikeCount' in statistic_response.json()['items'][0]['statistics']:     
        dislikeCount = statistic_response.json()['items'][0]['statistics']['dislikeCount']
    if 'commentCount' in statistic_response.json()['items'][0]['statistics']:     
        commentCount = statistic_response.json()['items'][0]['statistics']['commentCount'] 
    
    resultDataFrame  = resultDataFrame.append({'title':title,'youtubeTitle':youtubeTitle,
                                'viewCount_Youtube':viewCount, 'likeCount_Youtube':likeCount,
                                'dislikeCount_Youtube':dislikeCount, 'commentCount_Youtube':commentCount}, ignore_index = True)
    
jprint(search_response.json())

search_response.json()