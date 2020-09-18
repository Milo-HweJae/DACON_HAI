'''
test data.py
'''

import tensorflow as tf

import numpy as np
import pandas as pd

from pathlib import Path
import numpy as np
import pandas as pd


def load_csv(DATA_DIR, filename):
    df = pd.read_csv('/'.join([DATA_DIR, filename]), sep=',')
    data = df.iloc[:, 1:]
    data.index = df['time']
    data, index = preprocessing(data)
    
    return data, index

def preprocessing(data):

    index = data.index
    data = data.values
    
    return data, index

def split_target(data, is_train=0):
    # label = data[:, 79:]
    if is_train == 2:
        label = data[:, 79:]
        # label = data.drop([:, :79])
    else:
        label = []
    data = data[:, :79]
    return data, label

@tf.autograph.experimental.do_not_convert
def load_data(DATA_DIR, window_len, batch_size=64, is_train=0):
    if is_train == 0:
        file_names = ['train.csv']
    elif is_train == 1:
        file_names = ['test.csv']
    else:
        file_names = ['validation.csv']

    raw_data = []
    timestamp = []
    data, index = load_csv(DATA_DIR, file_names[0])
    raw_data.append(data)
    timestamp.append(index)
    
    BATCH_SIZE = batch_size
    BUFFER_SIZE = BATCH_SIZE * 10000
    
    ds_data = []
    ds_label =[]
    
    for data in raw_data:
        
        data, label = split_target(data, is_train)
        data = tf.data.Dataset.from_tensor_slices(data)
        
        if is_train == 2:
            label = tf.data.Dataset.from_tensor_slices(label)
            ds_label.append(label)
        ds_data.append(data)

    
    ds_data_tmp = []
    ds_label_tmp = []
    if is_train != 2:
        for sample_data in ds_data: # , sample_label , ds_label
            window_data = sample_data.window(window_len, shift=1, stride=1, drop_remainder=True)     # False:550800|448200 / True:550501|447901
            # window_label = sample_label.window(window_len, shift=1, stride=1, drop_remainder=True)
        
            window_data = window_data.flat_map(lambda x: x.batch(window_len))
            # window_label = window_label.flat_map(lambda x: x.batch(window_len))
            
            ds_data_tmp.append(window_data)
            # ds_label_tmp.append(window_label)
        
        del ds_data
        ds_data = ds_data_tmp[0]
        ds_data = ds_data.cache()
    else:
        for sample_data, sample_label in zip(ds_data, ds_label): #  
            window_data = sample_data.window(window_len, shift=1, stride=1, drop_remainder=True)     # False:550800|448200 / True:550501|447901
            window_label = sample_label.window(window_len, shift=1, stride=1, drop_remainder=True)
        
            window_data = window_data.flat_map(lambda x: x.batch(window_len))
            window_label = window_label.flat_map(lambda x: x.batch(window_len))
            
            ds_data_tmp.append(window_data)
            ds_label_tmp.append(window_label)
        
        del ds_data, ds_label #
        ds_data = ds_data_tmp[0]
        ds_label = ds_label_tmp[0]
        ds_data = ds_data.cache()    
        ds_label = ds_label.cache()
    

       
    if is_train != 2: # train dataset
        batch_data = ds_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return batch_data
    else: # test dataset (no shuffle)
        batch_data = ds_data.batch(BATCH_SIZE)
        batch_label = ds_label.batch(BATCH_SIZE)
        return batch_data, batch_label, timestamp 