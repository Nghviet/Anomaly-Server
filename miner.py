import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import association_rules
import itertools
import copy
import numpy as np
from scipy.stats import stats
import random
import pickle
from mlxtend.preprocessing import TransactionEncoder
from random import randrange
import warnings
import sys, os, time, math
warnings.filterwarnings('ignore')

TIME_WINDOW_IN_MINUTES = 1

def save_data(dataset, loc_file, pre_loc_file_url = '/content/drive/My Drive/hh101/'):
    loc_file = pre_loc_file_url + loc_file
    with open(loc_file, 'wb') as filehandle:
        pickle.dump(dataset, filehandle)
def load_data(loc_file, pre_loc_file_url = '/content/drive/My Drive/hh101/'):
    loc_file = pre_loc_file_url + loc_file
    with open(loc_file, 'rb') as filehandle:
        dataset = pickle.load(filehandle)
    return dataset

def remove_subset(list_of_list):              
    sets={frozenset(e) for e in list_of_list}  
    us=[]
    while sets:
        e=sets.pop()
        if any(e.issubset(s) for s in sets) or any(e.issubset(s) for s in us):
            continue
        else:
            us.append(sorted(list(e)))   
    return us

def print_list_with_minK(list_of_list, min_k_item):
    count = 0
    for item in list_of_list:
        if(len(item) >= min_k_item):
            count +=1
            print(item)
    print(f"Len -origin:{len(list_of_list)} ")
    print(f"Len -filter by min_k_item={min_k_item} :{count} ")

def is_binary_device_by_df(df):
    device_id = df['device_id']
    device_type = device_id.split("/")[0]
    if device_type == "micromotion" or device_type == "door":
        val = bool(True)
    else:
        val = bool(False)
    return val

def is_binary_device_by_id(device_id):
    if (device_id[0:1] == 'M' 
        or device_id[0:1] == 'D' 
        or device_id[0:2] == 'L0'):
        val = bool(True)
    else:
        val = bool(False)
    return val

def data_preprocessing(data):
    data['device_value'] =  data['device_value'].replace(['ON', 'OPEN'], 1)
    data['device_value'] =  data['device_value'].replace(['OFF', 'CLOSE'], 0)
    data.device_value = pd.to_numeric(data.device_value, errors='raise')
    data['is_binary_device'] = data.apply(is_binary_device_by_df, axis=1)
    data.drop(data[data['device_id'].astype(str).str[0:2] == 'BA'].index, axis =0, inplace = True)
    # data.drop(['ann'], axis=1, inplace =True)
    items = data['device_id'].unique()

def get_not_NaN(row):
    if row['light_sensors']:
        return row['light_sensors']
    elif row['temp_sensors']:
        return row['temp_sensors']
    return NaN
  
def discrete_data(data):
    # get list numeric devices
    dev_arr = data[data['is_binary_device'] == False]['device_id'].unique()
    labels = ['1','2','3','4', '5']
    data['mask'] = np.nan
    
    discrete_range = pd.DataFrame(columns=["device_id", "1","2","3","4","5"])
    
  # Discreate Tranform by device
    for dev in dev_arr:
        filter = data[data['device_id'] == dev]
        data['temp'] = pd.cut(filter['device_value'], bins=len(labels), labels=labels)
        data['mask'] = data['mask'].fillna(data['temp'])
        discrete_obj = {
            "device_id" : dev
        }
        for label in labels:
            discrete_obj[label] = data[data['mask'] == label][data['device_id'] == dev]['device_value'].max()
        print(discrete_obj)
        data.drop('temp', axis=1, inplace =True)
        discrete_range = discrete_range.append(discrete_obj, ignore_index=True)        
    return data, discrete_range

def convert_transaction_to_df_one_hot_vector(_transaction_dataset, sparse_format = True):
    te = TransactionEncoder()
    if(sparse_format):
        te_ary = te.fit(_transaction_dataset).transform(_transaction_dataset, sparse=True)
        return pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    te_ary = te.fit(_transaction_dataset).transform(_transaction_dataset)
    return pd.DataFrame(te_ary, columns=te.columns_)


def find_groups_dev(df_transaction_dataset, min_threshold = 0.9, max_len_itemset = 10, algo = 0, _verbose = 1, _low_meomory = False):
    df_transaction_dataset.iloc[:, :] = df_transaction_dataset.iloc[:,:].astype(bool)
    frequent_itemset = fpgrowth(df_transaction_dataset, min_support=min_threshold, use_colnames=True,max_len = max_len_itemset, verbose=_verbose)
    related_dev_groups = frequent_itemset[frequent_itemset['itemsets'].apply(lambda x: len(x) >= 1)]
    return related_dev_groups

def get_data(mongoClient,house_certificate_cn):
    user_id = house_certificate_cn.split('/')[0]
    house_id = house_certificate_cn.split('/')[1]

    print("Get data with userid:{}, houseid:{}".format(user_id, house_id))

    # data = pd.DataFrame(columns=["datetime", "device_id", "position_1", "position2", "device_value", "controller", "annotation"])
    
    db_data = []
    for item in mongoClient['development']['data'].find({
            "home": house_certificate_cn
        }):
        if item['valueKey'] == 'OperationStatus': continue
        obj = {
            'datetime' : datetime.datetime.fromtimestamp(int(item['timestamp'] / 1000)),
            'device_id': '{}/{}'.format(item['type'], item['device']),
            'position_1' : 'UNKNOWN',
            'position2' : 'UNKNOWN',
            'device_value' : item['value'],
            'controller' : item['device'].split('/')[0],
            'annotation' : 'UNKNOWN'
        }
        db_data.append(obj)
    data = pd.DataFrame(db_data)
    data = data[["datetime","device_id", "device_value"]]
    return data

def pattern_extraction(data): 
    print("Mining")
    state_transaction_dataset = []
    device_transaction_dataset = []
    number_of_hours =60*24 # ~ 8 weeks
    start_date = data.iloc[0].datetime
    last_datetime =  start_date + datetime.timedelta(hours=number_of_hours)
    data_to_make_group_dev = data[(data['datetime'] < last_datetime)].copy(deep=True)
    start = copy.deepcopy(start_date)
    num_window = 0

    while start<last_datetime:
        end = start + datetime.timedelta(minutes=(TIME_WINDOW_IN_MINUTES))
        data_extract = data_to_make_group_dev[(data_to_make_group_dev['datetime'] >= start) & (data_to_make_group_dev['datetime'] < end)]
        state_transaction_items = set()
        device_transaction_items = set()

        if not data_extract.empty:
            device_in_timewindow = data_extract['device_id'].unique()
            for device in device_in_timewindow:
                device_transaction_items.add(device)
                max_state = data_extract[data_extract['device_id'] == device]['mask'].max()
                if pd.isna(max_state):
                    max_state = "ON"
                state_transaction_items.add(device + '_' + str(max_state)) 

            # append current items to transaction db
            if(len(state_transaction_items)>1):
                num_window +=1
                state_transaction_dataset.append(list(state_transaction_items))
                device_transaction_dataset.append(list(device_transaction_items))
            if(len(state_transaction_items) == 0):
                print(f'transaction empty at {start} and {end}')
        start = end

    df_device_transaction_dataset = convert_transaction_to_df_one_hot_vector(device_transaction_dataset)
    df_state_transaction_dataset = convert_transaction_to_df_one_hot_vector(state_transaction_dataset)

    # Device frequent itemset
    FPOF_input_device = find_groups_dev(df_device_transaction_dataset,min_threshold=0.02, algo = -1, _verbose = 0)
    # Device state frequent itemset
    FPOF_input_state = find_groups_dev(df_state_transaction_dataset,min_threshold=0.02, algo = -1, _verbose = 0)

    # Device association rules
    device_assoc_rules = association_rules(FPOF_input_device, metric="confidence", min_threshold=0.9)
    # Device state association rules
    state_assoc_rules = association_rules(FPOF_input_state, metric="confidence", min_threshold=0.1)

    return FPOF_input_device, FPOF_input_state, device_assoc_rules, state_assoc_rules

def convert(frozen_set):
    s = ""
    print(frozen_set)
    for v in frozen_set:
        
        if len(s) == 0:
            s = v
        else:
            s = s + ";" + v
    return s

def process(mongoClient, house_certificate_cn):
    data = get_data(mongoClient, house_certificate_cn)
    data_preprocessing(data)
    data, discrete_range = discrete_data(data)

    FPOF_input_device, FPOF_input_state, device_assoc_rules, state_assoc_rules = pattern_extraction(data)
    state_assoc_rules['antecedents'] = state_assoc_rules['antecedents'].apply(convert)
    state_assoc_rules['consequents'] = state_assoc_rules['consequents'].apply(convert)
    house_certificate_cn = house_certificate_cn.replace("/","-")
    state_assoc_rules.to_csv('model/state_pattern/{}_state.csv'.format(house_certificate_cn), index = False, index_label = False)

    device_assoc_rules['antecedents'] = device_assoc_rules['antecedents'].apply(convert)
    device_assoc_rules['consequents'] = device_assoc_rules['consequents'].apply(convert)
    device_assoc_rules.to_csv('model/device_pattern/{}_device.csv'.format(house_certificate_cn), index = False, index_label = False)

    discrete_df = pd.DataFrame(discrete_range)
    discrete_df.to_csv('model/discrete/{}_discrete.csv'.format(house_certificate_cn), index = False, index_label = False)