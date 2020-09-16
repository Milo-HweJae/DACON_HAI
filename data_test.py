'''
test data.py
'''

import tensorflow as tf

import numpy as np
import pandas as pd

from pathlib import Path
import numpy as np
import pandas as pd

#print(tf.__version__)  # 2.2.0


def load_csv(DATA_DIR, filename):
    df = pd.read_csv('/'.join([DATA_DIR, filename]), sep=',')
    data = df.iloc[:, 1:]
    data.index = df['time']
    data, index = preprocessing(data)
    
    return data, index

def preprocessing(data):
    for key in data:
        MIN = 0
        MAX = 1
        if 'P1.' in key and ('CV' in key or ('B40' in key and '0B' not in key)) or ('P3.L' in key and '0' not in key) or ('P4.HT' in key and 'FD' not in key):
            MIN=0
            MAX=100
        elif 'P1' in key:
            if 'PIT' in key or 'B20' in key:
                MIN=0
                MAX=10
            elif 'FT' in key or 'B3005' in key or 'B400B' in key:
                if 'Z' in key:
                    MIN=0
                    MAX=3190
                else:
                    MIN=0
                    MAX=2500
            elif 'B3' in key or 'LIT' in key:
                MIN=0
                MAX=720
            elif 'TIT' in key:
                MIN=-50
                MAX=150
        elif 'P2' in key:
            if 'S' in key:
                MIN=0
                MAX=3600
            elif 'V' in key:
                if 'T01e' in key:
                    MIN=0
                    MAX=15
                elif '24' in key:
                    MIN=0
                    MAX=30
                else:
                    MIN=-10
                    MAX=10
        elif ('P3' in key and 'LC' in key) or ('P4' in key and 'T01' in key):
            MIN=0
            MAX=27648
        elif 'P3.' in key:
            MIN=0
            MAX=90
        else:
            if '.LD' in key:
                MIN=200
                MAX=600
            elif '_FD' in key:
                MIN = -0.01
                MAX = 0.01
            elif '_PS' in key:
                MIN=0
                MAX=450
            elif not 'attack' in key:
                MIN=200
                MAX=450

        data[key] = (data[key] - MIN) / (MAX - MIN)

    index = data.index
    data = data.values
    
    return data, index

def split_target(data):
    # label = data[:, 79:]
    data = data[:, :79]
    
    return data
# , label

# def filtering_attack(data, index):
#     # print(index[70285], index[70672], index[156685], index[157074])   
#     data1 = data[:70280,:]
#     data2 = data[70675:156680,:]
#     data3 = data[157080:,:]
    
#     index1 = index[:70280]
#     index2 = index[70675:156680]
#     index3 = index[157080:]
#     # print(data1, data2, data3)

#     return [data1, data2, data3], [index1, index2, index3]

@tf.autograph.experimental.do_not_convert
def load_data(DATA_DIR, window_len, batch_size=64, is_train=True):
    if is_train == True:
        file_names = ['train.csv']
    else:
        file_names = ['test.csv']

    raw_data = []
    timestamp = []
    data, index = load_csv(DATA_DIR, file_names[0])
    raw_data.append(data)
    timestamp.append(index)
    # data, index = load_csv(DATA_DIR, file_names[1])
    # raw_data.append(data)
    # timestamp.append(index)
    # data, index = load_csv(DATA_DIR, file_names[2])
    # raw_data.append(data)
    # timestamp.append(index)
    # if is_train == False:
    #     data, index = load_csv(DATA_DIR, file_names[0])
    #     raw_data.append(data)
    #     timestamp.append(index)
        
    

    num_sample = int((len(data) - window_len) * 0.01)
    # print(filename, num_sample)
    
    BATCH_SIZE = batch_size
    BUFFER_SIZE = BATCH_SIZE * 10000
    
    ds_data = []
    # ds_label =[]
    
    for data in raw_data:
        data = split_target(data)
        # , label 
        data = tf.data.Dataset.from_tensor_slices(data)
        # label = tf.data.Dataset.from_tensor_slices(label)
        
        ds_data.append(data)
        # ds_label.append(label)
        
        # if filename == file_names[0]:
        #     data1 = data
        #     label1 = label
        # else:
        #     data2 = data
        #     label2 = label
        
                
    # sample_data = sample_data1.concatenate(sample_data2)
    # sample_label = sample_label1.concatenate(sample_label2)

    
    ds_data_tmp = []
    # ds_label_tmp = []
    
    for sample_data in ds_data: # , sample_label , ds_label
        window_data = sample_data.window(window_len, shift=1, stride=1, drop_remainder=True)     # False:550800|448200 / True:550501|447901
        # window_label = sample_label.window(window_len, shift=1, stride=1, drop_remainder=True)
    
        window_data = window_data.flat_map(lambda x: x.batch(window_len))
        # window_label = window_label.flat_map(lambda x: x.batch(window_len))
        
        ds_data_tmp.append(window_data)
        # ds_label_tmp.append(window_label)
    
    del ds_data # , ds_label
    
    ds_data = ds_data_tmp[0]
    # for ds in ds_data_tmp[1:]:
    #     ds_data = ds_data.concatenate(ds)
    # ds_label = ds_label_tmp[0]
    # for label in ds_label_tmp[1:]:
    #     ds_label = ds_label.concatenate(label)
    # ds_data = ds_data_tmp[0].concatenate(ds_data_tmp[1])
    # ds_label = ds_label_tmp[0].concatenate(ds_label_tmp[1]) 
    
    ds_data = ds_data.cache()    
    # ds_label = ds_label.cache()
       
    if is_train==True: # train dataset
        batch_data = ds_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        # batch_label = ds_label.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return batch_data # , batch_label
    else: # test dataset (no shuffle)
        batch_data = ds_data.batch(BATCH_SIZE)
        # batch_label = ds_label.batch(BATCH_SIZE)
        return batch_data, timestamp # , batch_label