

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:40:08 2020

@author: lenovo
"""

import pandas as pd 
import re
# =============================================================================
# dataset = pd.read_csv('hasileksperiment-withInstagram.csv')
# listactor = pd.read_csv('secondattemptactorig.csv')
# 
# 
# dataset['hashtagigactor'] = dataset['actors']
# dataset['hashtagigactor'] = dataset['hashtagigactor'] 
# 
# 
# 
# for i, row in dataset.iterrows():
#       first_actor = row.actors
#       first_actor = first_actor.split(',')[0] 
#       first_actor = first_actor.lower()
#       first_actor = first_actor.replace(' ', '')
#       first_actor = re.sub(r'[^\w\s]','',first_actor)
#       first_actor = '#'+first_actor
#       dataset.at[i,'hashtagigactor'] = first_actor
# new_dataset = pd.merge(dataset, listactor , how ='left', on ='hashtagigactor')
# 
# 
# new_dataset.to_csv('hasileksperimen-WithInstagram2.csv')
# =============================================================================


dataset = pd.read_csv('hasileksperimen-WithInstagram2.csv') 

secondhashtag = pd.read_csv('instagramhashtagactor_secondscrappeddata.csv')
secondhashtag.rename(columns = {'Title': 'hashtagigactor2', 'Field1':'actorhashtagcount2' }, inplace= True)
thirdhashtag = pd.read_csv('instagramhashtagactor_thirdscrappeddata.csv') 
thirdhashtag.rename(columns = {'Title': 'hashtagigactor3', 'Field1': 'actorhashtagcount3'}, inplace = True)
fourthhashtag = pd.read_csv('instagramhashtagactor_fourscrappeddata.csv')
fourthhashtag.rename(columns = {'Title': 'hashtagigactor4', 'Field1': 'actorhashtagcount4'} ,inplace  = True)

# second hashtag 
dataset['hashtagigactor2'] = dataset['actors']
for i, row in dataset.iterrows():
       second_actor = row.actors
       second_actor = second_actor.split(',')[1] 
       second_actor = second_actor.lower()
       second_actor = second_actor.replace(' ', '')
       second_actor = re.sub(r'[^\w\s]','',second_actor)
       second_actor = '#'+second_actor
       dataset.at[i,'hashtagigactor2'] = second_actor
dataset= pd.merge(dataset, secondhashtag , how ='left', on ='hashtagigactor2')
# 

# third
dataset['hashtagigactor3'] = dataset['actors']
for i, row in dataset.iterrows():
       third_actor = row.actors
       third_actor = third_actor.split(',')[2] 
       third_actor = third_actor.lower()
       third_actor = third_actor.replace(' ', '')
       third_actor = re.sub(r'[^\w\s]','',third_actor)
       third_actor = '#'+third_actor
       dataset.at[i,'hashtagigactor3'] = third_actor
dataset= pd.merge(dataset, thirdhashtag , how ='left', on ='hashtagigactor3')

# fourth
dataset['hashtagigactor4'] = dataset['actors']
for i, row in dataset.iterrows():
    if(len(row.actors.split(',')) > 3):
       fourth_actor = row.actors
       fourth_actor = fourth_actor.split(',')[3] 
       fourth_actor = fourth_actor.lower()
       fourth_actor = fourth_actor.replace(' ', '')
       fourth_actor = re.sub(r'[^\w\s]','',fourth_actor)
       fourth_actor = '#'+fourth_actor
    dataset.at[i,'hashtagigactor4'] = fourth_actor
dataset= pd.merge(dataset, fourthhashtag , how ='left', on ='hashtagigactor4') 




dataset.to_csv('hasilEksperimen_FinalForm.csv')

