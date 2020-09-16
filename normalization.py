from pathlib import Path
import numpy as np
import pandas as pd


TRAIN_DATASET = sorted([x for x in Path("./dataset/train/").glob("*.csv")])
# TRAIN_DATASET

TEST_DATASET = sorted([x for x in Path("./dataset/test/").glob("*.csv")])
# TEST_DATASET

VALIDATION_DATASET = sorted([x for x in Path("./dataset/validation/").glob("*.csv")])
# VALIDATION_DATASET

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
# TRAIN_DF_RAW

TIMESTAMP_FIELD = "time"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"
VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])
# VALID_COLUMNS_IN_TRAIN_DATASET

TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

boundary_check(TRAIN_DF)

VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
# VALIDATION_DF_RAW
VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])
boundary_check(VALIDATION_DF)

TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
boundary_check(TEST_DF)

TRAIN_DF.to_csv('./dataset/train.csv')
TEST_DF.to_csv('./dataset/test.csv')
VALIDATION_DF.to_csv('./dataset/validation.csv')

    # -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:19:40 2020

@author: 111B-1
"""

