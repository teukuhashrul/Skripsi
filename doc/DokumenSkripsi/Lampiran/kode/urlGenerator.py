# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:37:15 2020

@author: lenovo
"""
import pandas as pd 
import re

# =============================================================================
# 
# # urltemplate = https://www.instagram.com/explore/tags/<hashtagname>/?hl=en
# all_url = ''
# for title in dataset.title: 
#     cleaned = re.sub(r'[^\w\s]','',title)
#     cleaned = cleaned.replace(' ', '')  
#     all_url = all_url + ' https://www.instagram.com/explore/tags/'+cleaned+'/ \n'
# 
# =============================================================================
# =============================================================================
# all_url = ''
# for actors in dataset.actors:
#     firstactor = actors.split(',')[0]
#     firstactor = firstactor.replace(' ', '')
#     firstactor = re.sub(r'[^\w\s]','',firstactor)
#     firstactor = firstactor.lower()
#     all_url = all_url + ' https://www.instagram.com/explore/tags/'+firstactor+'/ \n'
#     print(firstactor)
# =============================================================================
    
# second Actor    
dataset = pd.read_csv('hasileksperimen-WithInstagram2.csv')
all_url = ''
for actors in dataset.actors:
    firstactor = actors.split(',')[1]
    firstactor = firstactor.replace(' ', '')
    firstactor = re.sub(r'[^\w\s]','',firstactor)
    firstactor = firstactor.lower()
    all_url = all_url + ' https://www.instagram.com/explore/tags/'+firstactor+'/ \n'
    print(firstactor)
    
# third actor
dataset = pd.read_csv('hasileksperimen-WithInstagram2.csv')
all_url = ''
for actors in dataset.actors:
    firstactor = actors.split(',')[2]
    firstactor = firstactor.replace(' ', '')
    firstactor = re.sub(r'[^\w\s]','',firstactor)
    firstactor = firstactor.lower()
    all_url = all_url + ' https://www.instagram.com/explore/tags/'+firstactor+'/ \n'
    print(firstactor)
    
    
#fourthactor 
dataset = pd.read_csv('hasileksperimen-WithInstagram2.csv') 
all_url = ''
for actors in dataset.actors:
    splitted = actors.split(',')
    if(len(splitted) > 3):
        firstactor = actors.split(',')[3]
        firstactor = firstactor.replace(' ', '')
        firstactor = re.sub(r'[^\w\s]','',firstactor)
        firstactor = firstactor.lower()
        all_url = all_url + ' https://www.instagram.com/explore/tags/'+firstactor+'/ \n'
        print(firstactor)
