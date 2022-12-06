import pandas as pd
import torch
import numpy as np
import copy
import random
import argparse
import pickle

def getdataset(dataset,ratio):
    if(dataset == '5tasks'):
        print('load',dataset)
        if(ratio == 'smalldata'):
            x = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/CV_results/5tasks_mysmalldata.csv', sep = ' ', header = None))
            testx = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/CV_results/5tasks_mysmalldata_test.csv', sep = ' ', header = None))

        if(ratio == 'smallnet'):
            x = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/CV_results/results.csv', sep = ' ', header = None))
            testx = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/CV_results/results_test.csv', sep = ' ', header = None))

        if(ratio == 'fulldata'):
            x = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/CV_5tasks/5tasksfull_relative.csv', sep = ' ', header = None))
            testx = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/CV_5tasks/5tasksfull_relative_test.csv', sep = ' ', header = None))
            
    if(dataset == 'mimic5tasks'):
        print('load',dataset)
        if(ratio == '0.1'):
            x = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_validy01.csv', header = None))
            testx = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_x.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_testy01.csv', header = None))

        if(ratio == '0.5'):
            x = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_validy05.csv', header = None))
            testx = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_x.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_testy05.csv', header = None))

        if(ratio == '1'):
            x = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_validy10.csv', header = None))
            testx = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_x.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/mimic5tasks/5tasks_testy10.csv', header = None))

    if(dataset == '27tasksaddpair'):
        print('load',dataset)
        if(ratio == '0.1'):
            x = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_validx01_clean.csv',header = None,sep=' '))
            y = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_validy01_clean.csv', header = None,sep=' '))
            testx = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_testx01_clean.csv',header = None,sep=' '))
            testy = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_testy01_clean.csv', header = None,sep=' '))

        if(ratio == '0.5'):
            x = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_validx05_clean.csv',header = None,sep=' '))
            y = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_validy05_clean.csv', header = None,sep=' '))
            testx = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_testx05_clean.csv',header = None,sep=' '))
            testy = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_testy05_clean.csv', header = None,sep=' '))
        if(ratio == '1'):
            x = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_validx10_clean.csv',header = None,sep=' '))
            y = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_validy10_clean.csv', header = None,sep=' '))
            testx = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_testx10_clean.csv',header = None,sep=' '))
            testy = np.array(pd.read_csv('./gain_data/3000addpair/27tasks_testy10_clean.csv', header = None,sep=' '))

    if(dataset == '28tasks'):
        print('load',dataset)
        if(ratio == '0.1'):
            x = np.array(pd.read_csv('./gain_data/28tasks/28tasks_0.1x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/28tasks/3000samples_validy0.1.csv', header = None))
            testx = np.array(pd.read_csv('./gain_data/28tasks/3000samples_testx0.1.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/28tasks/3000samples_testy0.1.csv', header = None))
        # if(ratio == '0.2'):
        #     x = np.array(pd.read_csv('./gain_data/28tasks/28tasks_0.2x.csv',header = None))
        #     y = np.array(pd.read_csv('./gain_data/28tasks/3000samples_validy0.2.csv', header = None))
        #     testx = np.array(pd.read_csv('./gain_data/28tasks/28tasks_0.2x.csv',header = None))
        #     testy = np.array(pd.read_csv('./gain_data/28tasks/3000samples_validy0.2.csv', header = None))
        if(ratio == '0.5'):
            x = np.array(pd.read_csv('./gain_data/28tasks/28tasks_0.5x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/28tasks/3000samples_validy0.5.csv', header = None))
            testx = np.array(pd.read_csv('./gain_data/28tasks/3000samples_testx0.5.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/28tasks/3000samples_testy0.5.csv', header = None))
        if(ratio == '1'):
            x = np.array(pd.read_csv('./gain_data/28tasks/28tasks_1x.csv',header = None))
            y = np.array(pd.read_csv('./gain_data/28tasks/3000samples_validy1.csv', header = None))
            testx = np.array(pd.read_csv('./gain_data/28tasks/3000samples_testx1.csv',header = None))
            testy = np.array(pd.read_csv('./gain_data/28tasks/3000samples_testy1.csv', header = None))
    return x,y,testx,testy
    # if(dataset == '8tasks'):
    #     if(ratio == '0.1'):
    #         x = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
    #         y = np.array(pd.read_csv('./gain_data/CV_results/results_small_data.csv', sep = ' ', header = None))
    #         testx = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
    #         testy = np.array(pd.read_csv('./gain_data/CV_results/results_small_data.csv', sep = ' ', header = None))

    #     if(ratio == '0.5'):
    #         x = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
    #         y = np.array(pd.read_csv('./gain_data/CV_results/results.csv', sep = ' ', header = None))
    #         testx = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
    #         testy = np.array(pd.read_csv('./gain_data/CV_results/results_test.csv', sep = ' ', header = None))

    #     if(ratio == '1'):
    #         x = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
    #         y = np.array(pd.read_csv('./gain_data/CV_results/results_large_valid.csv', sep = ' ', header = None))
    #         testx = np.array(pd.read_csv('./gain_data/CV_results/cv0.1x.csv',header = None))
    #         testy = np.array(pd.read_csv('./gain_data/CV_results/results_large_test.csv', sep = ' ', header = None))