import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def _get_user_id(dataset_table):
    return list(set(dataset_table['user_id']))

def _get_group_id(dataset_table):
    return list(set(dataset_table['group']))

def _get_user_data(dataset_table, select_type, user_id, group, max_time):
    assert select_type == 'all' or select_type == 'group' or select_type == 'user'
    if select_type == 'all':
        dataset_filtered = np.array(dataset_table[(dataset_table['resptime'] < max_time)])
    elif select_type == 'group':
        dataset_filtered = np.array(dataset_table[(dataset_table['group']==group) & (dataset_table['resptime'] < max_time)])
    else:
        dataset_filtered = np.array(dataset_table[(dataset_table['user_id']==user_id) & (dataset_table['resptime'] < max_time)])
    dataset_table = pd.DataFrame(dataset_filtered,columns=dataset_table.columns.values)
    return dataset_table

def _get_epi_num(dataset_table, select_type, user_id, group, max_time):
    dataset_table = _get_user_data(dataset_table, select_type, user_id, group, max_time)
    return dataset_table.shape[0]

def _get_user2group_dict(dataset_table):
    group_all = _get_group_id(dataset_table)
    user_all = _get_user_id(dataset_table)
    group_dict = {}
    for i,user_id in enumerate(user_all):
        data_user = dataset_table[dataset_table['user_id']==user_id]
        group_dict[user_id] = data_user['group'].values[0]
    return group_dict

def _get_group_from_user(user_arr, group_dict, dim = 2):
    group_arr = []
    for i in range(len(user_arr)):
        group_arr.append(group_dict[user_arr[i]])
    group_arr = np.array(group_arr)
    if dim == 2:
        group_arr = group_arr.reshape((len(group_arr),1))
    return group_arr

def _get_group_users(dataset_path):
    raw_dataset = pd.read_csv(dataset_path)
    group_all = list(set(raw_dataset['group']))
    group_user_dict = {}
    for i,group_name in enumerate(group_all):
        data_each_group = raw_dataset[raw_dataset['group']==group_name]
        group_user_dict[group_name] = list(set(data_each_group['user_id']))
    return group_user_dict