# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:39:34 2020

@author: 111B-1
"""
import pickle

with open('atk_range.pkl','wb') as f:
    data = [[2112, 2303, 8892], [8892, 8989, 14352], [14352, 14541, 19267], [19267, 19326, 21802], [21802, 21890, 24890]]
    pickle.dump(data,f)