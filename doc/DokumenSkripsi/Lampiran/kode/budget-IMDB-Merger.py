# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:52:42 2020

@author: Teuku Hashrul
"""

import pandas as pd 


#read the original cleaned dataset 
dataset=pd.read_csv('hasileksperimen2.csv')
budget_imdbapi = pd.read_excel('scrappedBudgetFromIMDB-API_cleaned.xlsx') 
merged_dataset = pd.merge(dataset , budget_imdbapi , how='inner', on='title') 
merged_dataset.to_csv('hasileksperimen-withBudget.csv')     



