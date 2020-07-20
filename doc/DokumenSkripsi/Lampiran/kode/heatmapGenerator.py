# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:49:36 2020

@author: lenovo
"""

import pandas as pd 
import seaborn as sns 


dataset = pd.read_csv('hasileksperimen-WithInstagram2.csv')

corr = dataset[['votes', 'runtime', 'rating', 'metascore', 
                'profit', 'roi', 'us_budget', 'revenue',
                'viewCount', 'hashtagcount','actorhashtagcount']].corr()
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
    

        