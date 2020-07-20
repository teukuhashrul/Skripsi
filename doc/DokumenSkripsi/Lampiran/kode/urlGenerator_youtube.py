# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:43:07 2020

@author: lenovo
"""

import pandas as pd 
import seaborn as sns

dataset = pd.read_csv('hasileksperiment-withROI.csv') 


text = ''
for title in dataset.title:
    youtube_url = 'www.youtube.com/results?search_query=' + title + ' trailer'
    text = text + youtube_url + ' \n '




final_Data = pd.read_excel('hasileksperiment-withYoutube.xlsx')

corr = final_Data[['revenue', 'profit', 'roi','votes','metascore', 'runtime',
                'rating', 'us_budget', 'viewCount']].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
        
        

        