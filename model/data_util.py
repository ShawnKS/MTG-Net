import pandas as pd
import torch
import numpy as np
import copy
import random
import argparse
import pickle

def getdataset(dataset,ratio):
    print('load', dataset)
    if(dataset == 'ETTm1'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm1/ETTm1_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm1/ETTm1_val_y.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm1/ETTm1_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm1/ETTm1_y.csv',sep=' ',header=None))
    if(dataset == 'weather'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/weather/weather_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/weather/weather_val_y.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/weather/weather_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/weather/weather_y.csv',sep=' ',header=None))
    if(dataset == 'ETTm2'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm2/ETTm2_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm2/ETTm2_val_y.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm2/ETTm2_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/ETT/ETTm2/ETTm2_y.csv',sep=' ',header=None))
    if(dataset == 'illness'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/illness/illness_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/illness/illness_val_y.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/illness/illness_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/illness/illness_y.csv',sep=' ',header=None))
    if(dataset == 'chembl_small'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_small/chembl20_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_small/chembl20_valid.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_small/chembl20_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_small/chembl20_test.csv',sep=' ',header=None))
    if(dataset == 'chembl_13'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_13/chembl13_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_13/chembl13_val_y.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_13/chembl13_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_13/chembl13_y.csv',sep=' ',header=None))
    if(dataset == 'chembl_17'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_17/chembl17_mask.csv',sep=' ',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_17/chembl17_val_y.csv',sep=' ',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_17/chembl17_mask.csv',sep=' ',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MultiTaskGrouping/MTG-Net/chembl_17/chembl17_y.csv',sep=' ',header=None))
    if(dataset == 'muv'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/muv_x.csv',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/muv.csv',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/muv_x.csv',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/muv.csv',header=None))

    if(dataset == 'tox21'):
        x = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/tox21_x.csv',header=None))
        y = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/tox21.csv',header=None))
        testx = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/tox21_x.csv',header=None))
        testy = np.array(pd.read_csv('/home/covpreduser/Blob/v-xiaosong/projects/MTL/MTL4Drug/tox21.csv',header=None))

    if(dataset == 'taskonomy'):
        x = np.array(pd.read_csv('../gain_collection/collected_gain/CV_results/cv0.1x.csv',header = None))
        y = np.array(pd.read_csv('../gain_collection/collected_gain/CV_5tasks/5tasksfull_relative.csv', sep = ' ', header = None))
        testx = np.array(pd.read_csv('../gain_collection/collected_gain/CV_results/cv0.1x.csv',header = None))
        testy = np.array(pd.read_csv('../gain_collection/collected_gain/CV_5tasks/5tasksfull_relative.csv', sep = ' ', header = None))
            
    if(dataset == 'mimic5tasks'):
        x = np.array(pd.read_csv('../gain_collection/collected_gain/mimic5tasks/taskonomy_x.csv',header = None))
        y = np.array(pd.read_csv('../gain_collection/collected_gain/mimic5tasks/taskonomy_validy.csv', header = None))
        testx = np.array(pd.read_csv('../gain_collection/collected_gain/mimic5tasks/taskonomy_x.csv',header = None))
        testy = np.array(pd.read_csv('../gain_collection/collected_gain/mimic5tasks/taskonomy_testy.csv', header = None))

    if(dataset == 'mimic27'):
        x = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_x.csv',header = None,sep=' '))
        y = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_y_valid.csv', header = None,sep=' '))
        testx = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_x.csv',header = None,sep=' '))
        testy = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_y_test.csv', header = None,sep=' '))
    return x,y,testx,testy