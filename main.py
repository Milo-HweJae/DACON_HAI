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
from submission_csv import save_sub, read_sub

if __name__ == '__main__':
    DATA_DIR = './dataset' # path to data directory
    # TRAIN_LOG_DIR = './logs/train/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # MODEL_DIR = r'./model/' # path to model save directory
    
    LEN_TIME_WINDOW = 120 # history length (seconds)
    BATCH_SIZE = 64
    
    N_EPOCH = 10 # number of training epochs 
    ae_lr = 1e-5 # Aeutoencoder learning rate
    disc_lr = 1e-5 # Discriminator learning rate
    
    is_train = 0
    
    # prepare train dataset
    train_data = load_data(DATA_DIR, LEN_TIME_WINDOW, BATCH_SIZE, is_train=0) 
    
    # make a model
    m = model.cascaded_autoencoder(LEN_TIME_WINDOW, BATCH_SIZE)
    
    # train the model
    m.train(train_data, N_EPOCH, ae_lr, disc_lr)
    m.save() # save the model weights
    del train_data
    
    # Performance Evaluation
    m.restore() # restore the lastest model weights
    valid_data, valid_label, timestamp = load_data(DATA_DIR, LEN_TIME_WINDOW, BATCH_SIZE*2, is_train=2) 
    # y_pred_disc, y_pred_ae, y_true = pickle.load(open('./tr3_-mevaluate.pkl', 'rb'))
    
    y_pred_disc , y_pred_ae, y_true = m.evaluate(valid_data, valid_label)     # prediction으로 바꾸기
    # pickle.dump([y_pred_disc, y_pred_ae, y_pred_sub, y_true], open('./test1.pkl', 'wb'))
    #evaluate 함수 안에서 prediction 호출하면 predict 값이 나올꺼고 그걸 다시 p.result를 호출하여 결과가 나오게/// evaluate 호출하면 prediction 하고 result 실행   
    
    # y_pred_disc, y_pred_ae, y_pred_sub, y_true = pickle.load(open('./test1.pkl', 'rb'))
    y_pred_ae = y_pred_ae.reshape(43082,120,1)
    th_ae = 2e-4
    th_disc = 2e-4
    ylim_ae = (0.001, 1)
    ylim_disc = (0.1, 3)
    y_true = y_true[:,0]
    y_pred_ae = y_pred_ae[:,0]
    # 대충 멋진 결과 리포트 출력
    p.result(timestamp, y_pred_ae, y_pred_disc, y_true, th_ae, th_disc, ylim_ae, ylim_disc)
    
    test_data = load_data(DATA_DIR, LEN_TIME_WINDOW, BATCH_SIZE*2, is_train=1)
    th_ae = 2e-4
    th_disc = 1e-2 
    _ , y_pred_ae = m.evaluate(test_data)
    num = 358805*120 - 43042200
    y_pred_ae = y_pred_ae.tolist()
    for i in range(num): 
        y_pred_ae.insert(0,0)
    y_pred_ae = np.array(y_pred_ae)
    y_pred_ae = y_pred_ae.reshape(358805,120,1)
    y_pred_ae = y_pred_ae[:,0]
    y_pred_ae = y_pred_ae[:,0]
    data = read_sub()
    save_sub(data, y_pred_ae,th_ae)