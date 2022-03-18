import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, DistilBertModel, DistilBertConfig
from tqdm.notebook import tqdm
from torch.autograd import Variable
from datetime import datetime,timedelta
import time

def data_prepare(X, train_set=True, surge1=True):
    slp_ = X['slp']
    t_slp_ = X['t_slp']

    t_surge1_input_ = X['t_surge1_input']
    t_surge2_input_ = X['t_surge2_input']

    surge1_input_ = X['surge1_input']
    surge2_input_ = X['surge2_input']

    mean_surge1_input_ = np.mean(surge1_input_, axis=1)
    std_surge1_input_ = np.std(surge1_input_, axis=1)
    mean_surge2_input_ = np.mean(surge2_input_, axis=1)
    std_surge2_input_ = np.std(surge2_input_, axis=1)

    scaled_surge1_input_ = (surge1_input_ - mean_surge1_input_[:,None]) / std_surge1_input_[:,None]
    scaled_surge2_input_ = (surge2_input_ - mean_surge2_input_[:,None]) / std_surge2_input_[:,None]

    t_surge1_output_ = X['t_surge1_output']
    t_surge2_output_ = X['t_surge2_output']

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def hour_rounder(t):
        return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute//30))

    def time_to_hour(array):
        hours_array = np.empty_like(array)
        for i, times in enumerate(array):
            for j, t in enumerate(times):
                if t<0:
                    tt = (datetime(1970,1,1) + timedelta(seconds=int(t))).timetuple()
                else:
                    tt = (datetime.fromtimestamp(int(t))).timetuple() 
                hours_array[i][j] = tt.tm_yday * 24 + tt.tm_hour
        return hours_array / (366 * 24)

    pressures_same_time_2 = np.empty((*(t_surge2_input_.shape), 41, 41))
    for i, time_series in enumerate(t_surge2_input_):
        for j, time in enumerate(time_series):
            idx = find_nearest(t_slp_[i,:].flatten(), time)
            pressures_same_time_2[i, j, :, :] = slp_[i, idx, :, :]

    pressures_same_time_1 = np.empty((*(t_surge1_input_.shape), 41, 41))
    for i, time_series in enumerate(t_surge1_input_):
        for j, time in enumerate(time_series):
            idx = find_nearest(t_slp_[i,:].flatten(), time)
            pressures_same_time_1[i, j, :, :] = slp_[i, idx, :, :]

    mean_pressures_same_time_1 = np.mean(pressures_same_time_1, axis=(1,2,3))
    std_pressures_same_time_1 = np.std(pressures_same_time_1, axis=(1,2,3))
    mean_pressures_same_time_2 = np.mean(pressures_same_time_2, axis=(1,2,3))
    std_pressures_same_time_2 = np.std(pressures_same_time_2, axis=(1,2,3))

    scaled_pressures_same_time_1 = (pressures_same_time_1 - mean_pressures_same_time_1[:,None, None, None]) / std_pressures_same_time_1[:,None, None, None]
    scaled_pressures_same_time_2 = (pressures_same_time_2 - mean_pressures_same_time_2[:,None, None, None]) / std_pressures_same_time_2[:,None, None, None]

 
    hours_in_year_surge_1_ = time_to_hour(t_surge1_input_)
<<<<<<< Updated upstream
    hours_in_year_surge_2_ = time_to_hour(t_surge2_input_)
=======
    hours_in_year_surge_2_ = time_to_hour(t_surge1_input_)
>>>>>>> Stashed changes
    hours_in_year_surge_1_output_ = time_to_hour(t_surge1_output_)
    hours_in_year_surge_2_output_ = time_to_hour(t_surge2_output_)
    hours_in_year_slp_ = time_to_hour(t_slp_)
    hours_in_year_slp_ = time_to_hour(t_slp_)

    if surge1==True:
        if train_set == True:
            Y_train = pd.read_csv('./data/Y_train_surge.csv')
            Y_1 = Y_train[['surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4', 'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9']].to_numpy()
            Y_2 = Y_train[['surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4', 'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9']].to_numpy()


            datalen = len(surge1_input_)
            trainlen = int(0.9 * datalen)
            vallen = datalen - trainlen
            train_idx, val_idx = torch.utils.data.random_split(np.arange(datalen), [trainlen, vallen])

            pressure1_train, pressure1_val = scaled_pressures_same_time_1[train_idx], scaled_pressures_same_time_1[val_idx]
            surge1_train, surge1_val = surge1_input_[train_idx], surge1_input_[val_idx]
            t_surge1_train, t_surge1_val = hours_in_year_surge_1_[train_idx], hours_in_year_surge_1_[val_idx]
            Y_1_train, Y_1_val = Y_1[train_idx], Y_1[val_idx]

            train_data = list(zip(pressure1_train, surge1_train, t_surge1_train, Y_1_train))
            val_data = list(zip(pressure1_val, surge1_val, t_surge1_val, Y_1_val))

            batch_size = 8

            train_dataloader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True
            )

            val_dataloader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False
            )

            return train_dataloader, val_dataloader

        elif train_set == False:

            datalen = len(surge1_input_)

            pressure1_test = pressures_same_time_1
            surge1_test= surge1_input_
            t_surge1_test = hours_in_year_surge_1_

            test_data = list(zip(pressure1_test, surge1_test, t_surge1_test))

            batch_size = 509

            test_dataloader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )

            return test_dataloader
    
    if surge1==False:
        if train_set == True:
            Y_train = pd.read_csv('./data/Y_train_surge.csv')
            Y_1 = Y_train[['surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4', 'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9']].to_numpy()
            Y_2 = Y_train[['surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4', 'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9']].to_numpy()


            datalen = len(surge2_input_)
            trainlen = int(0.9 * datalen)
            vallen = datalen - trainlen
            train_idx, val_idx = torch.utils.data.random_split(np.arange(datalen), [trainlen, vallen])

            pressure2_train, pressure2_val = scaled_pressures_same_time_2[train_idx], scaled_pressures_same_time_2[val_idx]
            surge2_train, surge2_val = surge2_input_[train_idx], surge2_input_[val_idx]
            t_surge2_train, t_surge2_val = hours_in_year_surge_2_[train_idx], hours_in_year_surge_2_[val_idx]
            Y_2_train, Y_2_val = Y_2[train_idx], Y_2[val_idx]

            train_data = list(zip(pressure2_train, surge2_train, t_surge2_train, Y_2_train))
            val_data = list(zip(pressure2_val, surge2_val, t_surge2_val, Y_2_val))

            batch_size = 8

            train_dataloader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True
            )

            val_dataloader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False
            )

            return train_dataloader, val_dataloader

        elif train_set == False:
            datalen = len(surge2_input_)

            pressure2_test = pressures_same_time_2
            surge2_test= surge2_input_
            t_surge2_test = hours_in_year_surge_2_

            test_data = list(zip(pressure2_test, surge2_test, t_surge2_test))

            batch_size = 509

            test_dataloader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )

            return test_dataloader