# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:02:53 2020
@author: gnos
"""

# import packages
import model # 모델 정의된 코드 
from data import load_data # prepare dataset
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import plot as p

if __name__ == '__main__':
    DATA_DIR = r'../dataset' # path to data directory
    # TRAIN_LOG_DIR = './logs/train/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # MODEL_DIR = r'./model/' # path to model save directory
    
    LEN_TIME_WINDOW = 120 # history length (seconds)
    BATCH_SIZE = 64
    
    N_EPOCH = 40 # number of training epochs 
    ae_lr = 1e-5 # Aeutoencoder learning rate
    disc_lr = 1e-5 # Discriminator learning rate
    
    # prepare train dataset
    train_data, _ = load_data(DATA_DIR, LEN_TIME_WINDOW, BATCH_SIZE, is_train=True) 
    
    # make a model
    m = model.cascaded_autoencoder(LEN_TIME_WINDOW, BATCH_SIZE)
    
    # train the model
    # m.train(train_data, N_EPOCH, ae_lr, disc_lr)
    # m.save() # save the model weights
    # del train_data
    
    # Performance Evaluation
    m.restore() # restore the lastest model weights
    test_data, test_label, timestamp = load_data(DATA_DIR, LEN_TIME_WINDOW, BATCH_SIZE*2, is_train=False) 
    # y_pred_disc, y_pred_ae, y_true = pickle.load(open('./tr3_-mevaluate.pkl', 'rb'))
    
    
    y_pred_disc, y_pred_ae, y_pred_sub, y_true = m.evaluate(test_data, test_label)    
    pickle.dump([y_pred_disc, y_pred_ae, y_pred_sub, y_true], open('./test1.pkl', 'wb'))
       
    
    # y_pred_disc, y_pred_ae, y_pred_sub, y_true = pickle.load(open('./test1.pkl', 'rb'))
    
    
    th_ae = 2e-4
    th_disc = 0.92
    th_sub = [3e-4, 5e-6, 6e-6, 3e-5]
    
    ylim_ae = (1e-5, 5e-4)
    ylim_disc = (0.001, 1)
    
    # 대충 멋진 결과 리포트 출력
    p.result(timestamp, y_pred_disc, y_pred_ae, y_pred_sub, y_true, th_ae, th_sub, th_disc, ylim_ae, ylim_disc)
    