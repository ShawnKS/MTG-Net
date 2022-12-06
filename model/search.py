from itertools import combinations
import numpy as np
import pandas as pd
import pickle
import copy
import sys

def approx_optimal(sample_num,search_num,pred_traj,ground_truth,ground_truth_mask,dataset,mask=None):
    if(len(np.array(pred_traj).shape) == 3):
        total_gain_traj = []
        total_select = []
        for i in range(len(pred_traj)):
            pred_cost = pred_traj[i]
            pred_mask = mask[i]
            pred_cost[pred_mask==0] = -100
            # idx_list  = []
            # for mask_unit in ground_truth_mask:
            #     for j in range(len(pred_mask)):
            #         if((pred_mask[j] == mask_unit).all()):
            #             # print(mask_unit,NN_mask[i])
            #             idx_list.append(j)
            # pred_cost = pred_cost[idx_list]
            # pred_mask = pred_mask[idx_list]
            iterations = len(pred_cost[0])
            gain_traj = []
            sel_traj = []
            for iteration in range(1,iterations+1):
                rest_gain = copy.deepcopy(pred_cost)
                rest_mask = pred_mask
                # if(iteration == 1):
                #     print(rest_mask)
                # rest_mask = copy.deepcopy(ground_truth_mask)
                get_gain = copy.deepcopy(pred_cost)
                total_gain, selected_gr = approx_search(iteration,rest_gain, rest_mask, get_gain, pred_cost,sample_num,search_num,ground_truth,ground_truth_mask)
                sel_traj.append(selected_gr)
                gain_traj.append(total_gain)
            total_gain_traj.append(gain_traj)
            total_select.append(sel_traj)
            # print('*'*40)
            # print(total_select)
            # print(pred_cost)
        return total_gain_traj , total_select
    else:
        mask = copy.deepcopy(ground_truth_mask)
        pred_cost = pred_traj
        pred_cost[ground_truth_mask==0] = -99
        total_gain_traj = []
        total_select = []
        # search for multiple
        iterations = len(pred_cost[0])
        # chose_list = []
        for iteration in range(1,iterations+1):
            rest_gain = copy.deepcopy(pred_cost)
            rest_mask = copy.deepcopy(ground_truth_mask)
            get_gain = copy.deepcopy(pred_cost)
            total_gain, selected_gr = approx_search(iteration,rest_gain, rest_mask, get_gain, pred_cost,sample_num,search_num,ground_truth,ground_truth_mask)
            total_gain_traj.append(total_gain)
            total_select.append(selected_gr)
        return total_gain_traj,total_select



# approx_gain
def approx_search(iteration,rest_gain, rest_mask ,get_gain, pred_cost,sample_num,search_num,ground_truth,ground_truth_mask):
    total_list = [[] for i in range(search_num)]
    total_mask_list = [[] for i in range(search_num)]
    best_gains = np.zeros((search_num,len(pred_cost[0])))
    for j in range(iteration):
        chose_list = [] 
        mask_list = []
        best_gains = np.zeros((search_num,len(pred_cost[0])))
        initial_list = copy.deepcopy(rest_gain)
        initial_mask = copy.deepcopy(rest_mask)
        if(j == 0):
            for m in range(search_num):
                argmax_top_index = np.argmax(np.sum(initial_list,1))
                chose_list.append(initial_list[argmax_top_index])
                mask_list.append(initial_mask[argmax_top_index])
                initial_list = np.vstack((initial_list[:argmax_top_index], initial_list[argmax_top_index+1:]))
                initial_mask = np.vstack((initial_mask[:argmax_top_index], initial_mask[argmax_top_index+1:]))
            total_list = copy.deepcopy(chose_list)
            total_mask_list = copy.deepcopy(mask_list)
        else:
            for group in range(search_num):
                chose_list = np.array(rest_gain)
                mask_list = np.array(rest_mask)
                best_gain = np.zeros((len(pred_cost[0])))
                for k in range(len(pred_cost[0])):
                    if(j == 1):
                        best_gain[k] = np.array(total_list[group])[k]
                    else:
                        best_gain[k] = np.max(np.array(total_list[group])[:,k])
                get_gain = np.zeros(chose_list.shape)
                for l in range(len(best_gain)):
                    get_gain[:,l] = chose_list[:,l] - best_gain[l]
                get_gain[get_gain<0] = 0
                get_gains = np.sum(get_gain,1)
                _index = np.argmax(get_gains)
                if(get_gains[_index] > 0):
                    if(len(total_list[group].shape)==1):
                        total_list[group] = np.array([total_list[group], chose_list[_index]])
                        total_mask_list[group] = np.array([total_mask_list[group], mask_list[_index] ] )
                    else:
                        total_list[group] = np.vstack( (total_list[group], chose_list[_index]) )
                        total_mask_list[group] = np.vstack( (total_mask_list[group], mask_list[_index]) )
    total_gains = np.zeros((search_num))
    for i in range(len(total_list)):
        best_gain = np.zeros((len(pred_cost[0])))            
        for k in range(len(pred_cost[0])):
            if(j == 0):
                best_gain[k] = np.array(total_list[i])[k]
            else:
                best_gain[k] = np.max(np.array(total_list[i])[:,k])
        total_gains[i] = np.sum(best_gain)
    select_list = total_list[np.argmax(total_gains)]
    select_list_mask = total_mask_list[np.argmax(total_gains)]
    index_list = []
    if(j==0):
        index_list = []
    else:
        for _j in range(len(select_list_mask)):
            for i in range(len(ground_truth_mask)):
                if((ground_truth_mask[i] == select_list_mask[_j]).all()):
                    index_list.extend([i])
    selected_gr = ground_truth[index_list]
    total_gain = 0
    if(len(index_list) == 0):
        for i in range(len(ground_truth_mask)):
            if((ground_truth_mask[i] == select_list_mask).all()):
                selected_gr = ground_truth[i]
                total_gain = ground_truth[i].sum()
    else:
        for i in range(len(selected_gr[0])):
            total_gain = total_gain + selected_gr[np.argmax(select_list[:,i])][i]
            print('iteration',j,i,selected_gr[np.argmax(select_list[:,i])][i],'NN',select_list[np.argmax(select_list[:,i])][i],select_list[np.argmax(select_list[:,i])])
    return total_gain,selected_gr

def better(a, b):
    if(len(a) == 0):
        return b
    a_score = score_solution(a,a)
    b_score = score_solution(b,a)
    a_value = 0
    b_value = 0
    for i in a_score:
        a_value = a_value+i
    for i in b_score:
        b_value = b_value+i
    if(a_value > b_value):
        return a
    else:
        return b

def dominates(first, second):
    for i in range(len(first)):
        if(first[i] < second[i]):
            return False
    return True

def cfilter(candidates, best_score, budget):
    after_filter = []
    for candidate in candidates:
        if((candidate.cost <= budget) & (not dominates(best_score,candidate.performance))):
            after_filter.append(candidate)
    return after_filter

def score_solution(solution,candidates_in,size = -1):
    if(len(solution) == 0):
        score = -10000 * np.ones(len(candidates_in[0].performance),)
    else:
        score = -10000 * np.ones(len(solution[0].performance),)
    for i in solution:
        for j in range(len(solution[0].performance)):
            score[j] = max( i.performance[j], score[j] )
    return score

def get_sorting_score(running_score, a):
    amin = 1000000
    for i in range(len(running_score)):
        amin = min(amin, running_score[i]-a.performance[i])
    return amin

def best_search(candidates_in, budget, running_solution):
    if(len(candidates_in) == 0):
        return running_solution
    running_score = score_solution(running_solution, candidates_in,len(candidates_in[0].performance))
    candidates = cfilter(candidates_in, running_score,budget)
    if(len(candidates) == 0):
        return running_solution
    sorted(candidates,key=lambda x:get_sorting_score(running_score, x))
    best_solution = copy.deepcopy(running_solution)
    count = 0
    while(len(candidates)!=0):
        count = count + 1
        running_solution.append( candidates[-1] )
        candidates.pop()
        best_below = copy.deepcopy(best_search( candidates, budget-running_solution[-1].cost,running_solution ))
        running_solution.pop()
        best_solution = copy.deepcopy(better(best_solution, best_below))
    return best_solution