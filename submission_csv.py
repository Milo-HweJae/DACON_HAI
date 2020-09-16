# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:37:18 2020

@author: 111B-1
"""

import csv
def read_sub():
    f = open('./dataset/sample_submission.csv','r',encoding='utf-8')
    
    
    rdr = csv.reader(f)
    data=[]
    i=0
    for line in rdr:
        i = i + 1
        
        if i==1:
            continue
        data.append(line)
    
    f.close()
    
    return data

def save_sub(data, y_pred_ae):
    f_ = open('./dataset/submission.csv','w',encoding='utf-8',newline='')
    wr = csv.writer(f_)
    wr.writerow(['time', 'attack'])
    for i in range(len(data)):
        if i < 119:
            wr.writerow([data[i][0],data[i][1]])
            continue
        # if i >= len(y_pred_ae) + 238:
        #     wr.writerow([data[i][0],data[i][1]])
        #     continue
        # else:
        wr.writerow([data[i][0], y_pred_ae[i-119]])
    f_.close()
    