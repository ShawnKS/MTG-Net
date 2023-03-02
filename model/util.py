from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from MTGNet import *
import pandas as pd
import torch
import numpy as np
import copy
import random
import os
import argparse
import pickle
from data_util import *

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def unravel_index(flat_idx, shape): 
     multi_idx = [] 
     r = flat_idx 
     for s in shape[::-1]: 
         multi_idx.append(r % s) 
         r = r // s 
     return multi_idx[: :-1]



def softmax(x, temperature=1):
    return np.exp(x/temperature)/sum(np.exp(x/temperature))

def model_training(dataset, ratio,temperature, num_layers,num_hidden,ensemble_num,gpu_id,step=1,end_num=1,seed=1,strategy = 'active',dropout_rate = 0.5):
    collect_embedding = False
    task_embedding_list = []
    encoder_output_list = []
    encoder_mask_list = []
    print("current step is",step)
    print("current strategy is",strategy)
    print("current seed is", seed)
    device = 'cuda:'+ gpu_id
    print(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x,y,testx,testy = getdataset(dataset,ratio)
    import copy
    save_x = copy.deepcopy(x)
    task_id = torch.from_numpy(np.array(range(len(x[0])))).to(device)
    task_id_all = task_id.repeat(len(x),1)
    end_num = int((len(x[0]) * (len(x[0]) - 1 )  )/(2 * step))
    if( (dataset == '27tasks') & (strategy == 'active')):
        end_num = end_num + len(x[0]) - 1
    if( (dataset == '27tasksaddpair') & (strategy == 'active')):
        end_num = end_num + len(x[0]) - 1
    if(strategy == 'activeper'):
        end_num = end_num
        #  + len(x[0]) - 1
    if(strategy == 'active'):
        end_num = end_num + len(x[0]) - 1
    print(end_num)
    print('end iter',end_num)
    testx_align = testx
    testy_align = testy
    import copy
    total_x = copy.deepcopy(testx)
    total_y = copy.deepcopy(testy)
    active_iter_num = 1000000
    train_perf = np.array([])
    # step = 15
    valid_perf = np.array([])
    test_perf = np.array([])
    all_perf = np.array([])
    total_num = len(x)
    sample_list = []
    for i in range(len(x)):
        if((len(x[i][x[i]==1])==1)):
            sample_list.append(i)
        if((len(x[i][x[i]==1])==len(x[0]))):
            sample_list.append(i)
    mask = copy.deepcopy(x)
    print(sample_list)
    mask_test = copy.deepcopy(testx_align)
    X_select = x[ np.setdiff1d(np.array(range(len(x))), sample_list) ]
    y_select = y[ np.setdiff1d(np.array(range(len(y))), sample_list) ]
    mask_select = mask[np.setdiff1d(np.array(range(len(mask))), sample_list)]
    
    print(sample_list)
    X_train = x[sample_list]
    y_train = y[sample_list]
    mask_train = mask[sample_list]
    if((dataset != '27tasks')):
        test_list = np.array(range(len(testx_align)))
        test_list = np.setdiff1d(test_list, sample_list)
        X_test = testx_align[test_list]
        y_test = testy_align[test_list]
        mask_test = copy.deepcopy(X_test)
    else:
        X_test = testx_align
        y_test = testy_align
        mask_test = copy.deepcopy(X_test)

    # X_train, X_test, y_train, y_test = train_test_split(x,y,test_size =0.9)
    print('max y', np.max(y_train))
    print('min y', np.min(y_train))
# .to(device)
    X_train, X_test, X_select, y_train, y_test, y_select = torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.FloatTensor(X_select), torch.FloatTensor(y_train), torch.FloatTensor(y_test), torch.FloatTensor(y_select)
    mask_train, mask_test,mask_select = torch.FloatTensor(mask_train), torch.FloatTensor(mask_test), torch.FloatTensor(mask_select)
    batch_size = 128
    # X_valid,y_valid, mask_valid = torch.FloatTensor(X_valid).to(device), torch.FloatTensor(y_valid).to(device), torch.FloatTensor(mask_valid).to(device)
    epoch_num = 10000
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    ensemble_capacity = ensemble_num
    max_patience = 5
    total_run_train_loss = []
    total_run_valid_loss = []
    total_run_loss = []
    total_run_test_loss = []
    total_pred_traj = []
    total_mask_traj = []

    model = HOINetTransformer(num_layers=num_layers,
            model_dim=num_hidden,
            num_heads=1,
            task_dim = len(x[0]),
            ffn_dim= num_hidden,
            dropout=dropout_rate).to(device)
    for active_num in range(active_iter_num):
        ensemble_list = []
        ensemble_loss = np.array([])
        total_train_loss = np.array([])
        total_valid_loss = np.array([])
        total_test_loss = np.array([])
        train_perb = torch.randperm(len(X_train)).to(device)
        X_train, y_train, mask_train = X_train[train_perb], y_train[train_perb], mask_train[train_perb]
        # test_perb = torch.randperm(len(X_test)).to(device)
        # X_test, y_test, mask_test = X_test[test_perb], y_test[test_perb], mask_test[test_perb]
        # se_perb = torch.randperm(len(X_test)).to(device)
        # X_select, y_select, mask_select = X_select[test_perb], y_select[test_perb], mask_select[test_perb]
        min_loss = 100
        patience_loss = 100
        print('cost',active_num)
                # 0.005
        # retrain or not?
        model = HOINetTransformer(num_layers=num_layers,
                model_dim=num_hidden,
                num_heads=1,
                task_dim = len(x[0]),
                ffn_dim= num_hidden,
                dropout=dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr= 0.007,weight_decay=0.0005 )
        # print("current activa learning iteration is ", active_num)
        patience = 0
        # print(len(X_train) - 8)
        # print(len(X_test))
        for epoch in range(epoch_num):
            model.train()
            # train_perb = torch.randperm(len(X_train)).to(device)
            # X_train, y_train, mask_train = X_train[train_perb], y_train[train_perb], mask_train[train_perb]
            total_output = torch.Tensor([])
            ground_truth = torch.Tensor([])
            total_mask = torch.Tensor([])
            batch_size = 25000
            # model training, ensemble
            for i in range(int(len(X_train)/batch_size)+1):
                optimizer.zero_grad()
                batch_x = X_train[batch_size*i: batch_size*(i+1)].to(device)
                if(len(batch_x) == 0): break
                batch_mask = mask_train[batch_size*i: batch_size*(i+1)].to(device)
                batch_y = y_train[batch_size*i: batch_size*(i+1)].to(device)
                task_id_batch = task_id_all[:len(batch_x)].to(device)
                output,attentions,task_embedding,encoder_output = model(batch_x, task_id_batch)
                # print(encoder_output.cpu().detach().numpy())
                # print(encoder_output.cpu().detach().numpy().shape)
                # print(batch_mask.shape)
                # print(torch.mul(encoder_output.permute(2,0,1),batch_x).permute(1,2,0))
                encoder_output = torch.mul(encoder_output.permute(2,0,1),batch_x).permute(1,2,0)
                # encoder_output_list.append(encoder_output.cpu().detach().numpy())
                if(active_num == end_num):
                    encoder_output_list.append(encoder_output.cpu().detach().numpy())
                    task_embedding_list.append(task_embedding.cpu().detach().numpy())
                    np.save('./embedding_collect_0.1'+'/'+dataset+'_'+str(seed)+'_task_embedding_list_sd.npy',task_embedding_list)
                    np.save('./embedding_collect_0.1'+'/'+dataset+'_'+str(seed)+'_encoder_output_list_sd.npy',encoder_output_list)
                # print(output[:,:,0])
                total_output = torch.cat([total_output,output.cpu().detach()],0)
                ground_truth = torch.cat([ground_truth,batch_y.cpu().detach()],0)
                total_mask = torch.cat([total_mask,batch_mask.cpu().detach()],0)
                loss = criterion(output[batch_mask!=0], batch_y[batch_mask!=0])
                loss.backward()
                optimizer.step()
            train_loss = criterion(total_output[total_mask!=0], ground_truth.mul(total_mask)[total_mask!=0] ).cpu().detach().numpy()
            if(epoch < ensemble_capacity):
                tmp_model = copy.deepcopy(model)
                tmp_model.eval()
                ensemble_list.append(tmp_model)
                ensemble_loss = np.hstack((ensemble_loss, train_loss))
            else:
                if( train_loss < ensemble_loss.max()):
                    ensemble_loss[np.argmax(ensemble_loss)] = train_loss
                    tmp_model = copy.deepcopy(model)
                    tmp_model.eval()
                    ensemble_list[np.argmax(ensemble_loss)] = tmp_model
            total_train_loss = np.hstack((total_train_loss, train_loss))
            if(train_loss < patience_loss):
                patience = 0
                patience_loss = train_loss
            else:
                if(patience == 0):
                    tmp_model = copy.deepcopy(model)
                patience = patience + 1
                # print(patience)
            if(patience == max_patience):
                model = copy.deepcopy(tmp_model)
                break
            
            model.eval()
            batch_size = 25000
            ensemble_valid_output = torch.Tensor([])
            ensemble_ground_truth = torch.Tensor([])
            ensemble_total_mask = torch.Tensor([])
            ensemble_test_output = torch.Tensor([])
            ensemble_test_ground_truth = torch.Tensor([])
            ensemble_test_mask = torch.Tensor([])
            for ensemble_idx in range(len(ensemble_list)):
                ensemble_model = ensemble_list[ensemble_idx]
                total_output = torch.Tensor([])
                ground_truth = torch.Tensor([])
                total_mask = torch.Tensor([])
                for i in range(int(len(X_select)/batch_size)+1):
                    batch_x = X_select[batch_size*i: batch_size*(i+1)].to(device)
                    if(len(batch_x) == 0): break
                    batch_mask = mask_select[batch_size*i: batch_size*(i+1)].to(device)
                    batch_y = y_select[batch_size*i: batch_size*(i+1)].to(device)
                    task_id_batch = task_id_all[:len(batch_x)].to(device)
                    output,attentions,task_embedding,encoder_output = ensemble_model(batch_x, task_id_batch)
                    if(ensemble_idx==0):
                        encoder_output = torch.mul(encoder_output.permute(2,0,1),batch_x).permute(1,2,0)
                        if(active_num == end_num):
                            task_embedding_list.append(task_embedding.cpu().detach().numpy())
                            encoder_output_list.append(encoder_output.cpu().detach().numpy())
                            np.save('./embedding_collect'+'/'+dataset+'_'+str(seed)+'_task_embedding_list_sd.npy',task_embedding_list)
                            np.save('./embedding_collect'+'/'+dataset+'_'+str(seed)+'_encoder_output_list_sd.npy',encoder_output_list)
                    output = output.mul(batch_mask)
                    total_output = torch.cat([total_output,output.cpu().detach()],0)
                    ground_truth = torch.cat([ground_truth,batch_y.cpu().detach()],0)
                    total_mask = torch.cat([total_mask,batch_mask.cpu().detach()],0)
                if(len(ensemble_valid_output)== 0):
                    ensemble_valid_output = total_output.clone().unsqueeze(dim=0)
                else:
                    ensemble_valid_output = torch.cat([ensemble_valid_output, total_output.unsqueeze(dim=0) ])
                # ensemble_ground_truth = torch.stack([ensemble_ground_truth, ground_truth],1)
                # ensemble_total_mask = torch.stack([ensemble_total_mask, total_mask],1)
                
                test_output = torch.Tensor([])
                test_ground_truth = torch.Tensor([])
                test_mask = torch.Tensor([])
                for i in range(int(len(X_test)/batch_size)+1):
                    batch_x = X_test[batch_size*i: batch_size*(i+1)].to(device)
                    if(len(batch_x) == 0): break
                    batch_mask = mask_test[batch_size*i: batch_size*(i+1)].to(device)
                    batch_y = y_test[batch_size*i: batch_size*(i+1)].to(device)
                    task_id_batch = task_id_all[:len(batch_x)].to(device)
                    output,attentions,_,_ = ensemble_model(batch_x, task_id_batch)
                    output = output.mul(batch_mask)
                    test_output = torch.cat([test_output,output.cpu().detach()],0)
                    test_ground_truth = torch.cat([test_ground_truth,batch_y.cpu().detach()],0)
                    test_mask = torch.cat([test_mask,batch_mask.cpu().detach()],0)
                if(len(ensemble_test_output)== 0):
                    ensemble_test_output = test_output.clone().unsqueeze(dim=0)
                else:
                    ensemble_test_output = torch.cat([ensemble_test_output, test_output.unsqueeze(dim=0) ])
            valid_loss = criterion(torch.mean(ensemble_valid_output,dim=0)[total_mask!=0], ground_truth.mul(total_mask)[total_mask!=0] ).cpu().detach().numpy()
            test_loss = criterion(torch.mean(ensemble_test_output,dim=0)[test_mask!=0],test_ground_truth.mul(test_mask)[test_mask!=0] ).cpu().detach().numpy()
            total_valid_loss = np.hstack((total_valid_loss, valid_loss))
        
        total_run_train_loss.append(total_train_loss)
        total_run_valid_loss.append(total_valid_loss)
        total_run_test_loss.append(total_test_loss)
        
        total_output = torch.mean(ensemble_valid_output,dim=0)
        test_output = torch.mean(ensemble_test_output,dim=0)
        model.eval()
        valid_perf = np.hstack((valid_perf,valid_loss))
        train_perf = np.hstack((train_perf,train_loss))
        test_perf = np.hstack((test_perf, test_loss))
        print('train loss',train_perf)
        print('valid loss',valid_perf)
        print("test loss",test_perf)

        test_output = torch.mean(ensemble_test_output,dim=0)
        if(dataset=='27tasks'):
            model_pred = torch.cat([test_output,total_output,y_train.cpu().detach()],0).numpy()
            model_x = torch.cat([X_test.cpu().detach(),X_select.cpu().detach(), X_train.cpu().detach()],0).numpy()
        else:
            model_pred = torch.cat([test_output,y_train.cpu().detach()],0).numpy()
            model_x = torch.cat([X_test.cpu().detach(), X_train.cpu().detach()],0).numpy()
        print(model_x.shape)
        new_index =[]
        
        total_pred_traj.append(model_pred)
        total_mask_traj.append(model_x)
        path = './log/'+dataset+'/'+ ratio +'/'
        if not os.path.exists(path):
            os.makedirs(path)
        if(active_num >= end_num):
            temp = ''
            if(strategy != 'active'):
                temp = strategy
            with open('./log/'+dataset+'/'+ ratio +'/train_pertask_ensembleprob_loss_'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_num)+'.pkl', "wb") as fp:
                pickle.dump(total_run_train_loss, fp)
            with open('./log/'+dataset+'/'+ ratio +'/valid_pertask_ensembleprob_loss_'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_num)+'.pkl', "wb") as fp:
                pickle.dump(total_run_valid_loss, fp)
            with open('./log/'+dataset+'/'+ ratio +'/test_pertask_ensembleprob_loss_'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_num)+'.pkl', "wb") as fp:
                pickle.dump(total_run_test_loss, fp)
            with open('./log/'+dataset+'/'+ ratio +'/pred_pertask_traj'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_num)+'.pkl', "wb") as fp:
                pickle.dump(total_pred_traj, fp)
            with open('./log/'+dataset+'/'+ ratio +'/mask_pertask_traj'+temp+'_'+str(temperature)+'_'+str(seed)+'_'+str(num_layers)+'_'+str(active_num)+'.pkl', "wb") as fp:
                pickle.dump(total_mask_traj, fp)
            print('The step for saving is', active_num)
            return total_pred_traj

        if(strategy == 'activeper'):
            col = active_num % len(x[0])
            print(col)
            if(active_num < len(x[0])):
                sel_index = torch.argmax(abs(total_output[:,col]))
                X_train = torch.cat([X_train, X_select[sel_index].unsqueeze(0)],0)
                y_train = torch.cat([y_train, y_select[sel_index].unsqueeze(0)],0)
                mask_train = torch.cat([mask_train,mask_select[sel_index].unsqueeze(0)]  ,0)
                X_select = del_tensor_ele(X_select,sel_index)
                y_select = del_tensor_ele(y_select,sel_index)
                mask_select = del_tensor_ele(mask_select,sel_index)
                total_output = del_tensor_ele(total_output,sel_index)
                if(dataset != '27tasks'):
                    X_test = del_tensor_ele(X_test,sel_index)
                    y_test = del_tensor_ele(y_test,sel_index)
                    mask_test = del_tensor_ele(mask_test,sel_index)
                if(step == 2):
                    total_output[total_output==0]=100
                    sel_index = torch.argmin(abs(total_output[:,col]))
                    X_train = torch.cat([X_train, X_select[sel_index].unsqueeze(0)],0)
                    y_train = torch.cat([y_train, y_select[sel_index].unsqueeze(0)],0)
                    mask_train = torch.cat([mask_train,mask_select[sel_index].unsqueeze(0)]  ,0)
                    X_select = del_tensor_ele(X_select,sel_index)
                    y_select = del_tensor_ele(y_select,sel_index)
                    mask_select = del_tensor_ele(mask_select,sel_index)
                    total_output = del_tensor_ele(total_output,sel_index)
                    if(dataset != '27tasks'):
                        X_test = del_tensor_ele(X_test,sel_index)
                        y_test = del_tensor_ele(y_test,sel_index)
                        mask_test = del_tensor_ele(mask_test,sel_index)
            else:
                for i in range(step):
                    if(i>0):
                        active_num = active_num + 1
                    col = active_num % len(x[0])
                    sel_index = torch.argmax(abs(total_output[:,col]))
                    X_train = torch.cat([X_train, X_select[sel_index].unsqueeze(0)],0)
                    y_train = torch.cat([y_train, y_select[sel_index].unsqueeze(0)],0)
                    mask_train = torch.cat([mask_train,mask_select[sel_index].unsqueeze(0)]  ,0)
                    X_select = del_tensor_ele(X_select,sel_index)
                    y_select = del_tensor_ele(y_select,sel_index)
                    mask_select = del_tensor_ele(mask_select,sel_index)
                    total_output = del_tensor_ele(total_output,sel_index)
                    if(dataset != '27tasks'):
                        X_test = del_tensor_ele(X_test,sel_index)
                        y_test = del_tensor_ele(y_test,sel_index)
                        mask_test = del_tensor_ele(mask_test,sel_index)
                    if(step == 2):
                        total_output[total_output==0]=100
                        sel_index = torch.argmin(abs(total_output[:,col]))
                        X_train = torch.cat([X_train, X_select[sel_index].unsqueeze(0)],0)
                        y_train = torch.cat([y_train, y_select[sel_index].unsqueeze(0)],0)
                        mask_train = torch.cat([mask_train,mask_select[sel_index].unsqueeze(0)]  ,0)
                        X_select = del_tensor_ele(X_select,sel_index)
                        y_select = del_tensor_ele(y_select,sel_index)
                        mask_select = del_tensor_ele(mask_select,sel_index)
                        total_output = del_tensor_ele(total_output,sel_index)
                        if(dataset != '27tasks'):
                            X_test = del_tensor_ele(X_test,sel_index)
                            y_test = del_tensor_ele(y_test,sel_index)
                            mask_test = del_tensor_ele(mask_test,sel_index)
                        break