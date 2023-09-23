##################
# 4x-task
##################
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import cv2

###############
def data_load():
    # laod geo_data
    geo_data = pd.read_csv("huadong_geo_data.csv", encoding='gbk')
    geo_data = geo_data.sort_values(by=['lat', 'lon'], ascending=[False, True])
    geo_z = geo_data[['z']].to_numpy()
    print(geo_z.shape)
    geo_z_array = geo_z.reshape((162,112))
    ###############
    # load all data
    data_in = np.load('data_in.npy').astype(np.float32)
    data_out = np.load('data_out.npy').astype(np.float32)
    print(data_in.shape)
    print(data_out.shape)

    # normalize data
    data_all = np.concatenate((data_in,data_out), axis=2)
    index_idx = data_all.shape[0]
    
    maxs = data_all.max(axis=(0,2), keepdims=True)
    mins = data_all.min(axis=(0,2), keepdims=True)
    data_in_new = (data_in-mins)/(maxs-mins)
    data_out_new = (data_out-mins)/(maxs-mins)

    # scale factor data
    ################
    # geo data
    geo_z_array = geo_z.reshape((162,112))
    geo_z_array1 = cv2.resize(geo_z_array, (44, 64), interpolation=cv2.INTER_CUBIC)
    geo_z_array2 = cv2.resize(geo_z_array, (22, 32), interpolation=cv2.INTER_CUBIC)
    geo_z_array3 = cv2.resize(geo_z_array, (11, 16), interpolation=cv2.INTER_CUBIC)
    # axis=0,1,2
    geo_data_mid = np.zeros((index_idx,32,22))
    geo_data_lr = np.zeros((index_idx,16,11))

    for i in range(index_idx):
        geo_data_mid[i,:,:] = geo_z_array2
        geo_data_lr[i,:,:] = geo_z_array3

    geo_data_new_mid = (geo_data_mid - geo_data_mid.min()) / (geo_data_mid.max() - geo_data_mid.min())
    geo_data_new_lr = (geo_data_lr - geo_data_lr.min()) / (geo_data_lr.max() - geo_data_lr.min())

    print(geo_data_new_mid.shape, geo_data_new_mid.max(), geo_data_new_mid.min())
    print(geo_data_new_lr.shape, geo_data_new_lr.max(), geo_data_new_lr.min())

    ################
    # weather data
    def make_data(data1,data2):
        
        data_in = np.zeros((index_idx,64,44))
        data_out = np.zeros((index_idx,64,44))
        
        data_in_mid = np.zeros((index_idx,32,22))
        data_out_mid = np.zeros((index_idx,32,22))
        
        data_in_lr = np.zeros((index_idx,16,11))
        data_out_lr = np.zeros((index_idx,16,11))
        
        for i in range(data1.shape[0]):
            data1_reshape = data1[i:i+1,:].reshape((162,112))
            data2_reshape = data2[i:i+1,:].reshape((65,45))
            
            data_in_hr1 = cv2.resize(data1_reshape, (44, 64), interpolation=cv2.INTER_CUBIC)
            data_in_mid1 = cv2.resize(data1_reshape, (22, 32), interpolation=cv2.INTER_CUBIC)
            data_in_lr1 = cv2.resize(data1_reshape, (11, 16), interpolation=cv2.INTER_CUBIC)
            
            data_out_hr1 = cv2.resize(data2_reshape, (44, 64), interpolation=cv2.INTER_CUBIC)
            data_out_mid1 = cv2.resize(data2_reshape, (22, 32), interpolation=cv2.INTER_CUBIC)
            data_out_lr1 = cv2.resize(data2_reshape, (11, 16), interpolation=cv2.INTER_CUBIC)
            
            data_in[i,:,:] = data_in_hr1
            data_out[i,:,:] = data_out_hr1
            
            data_in_mid[i,:,:] = data_in_mid1
            data_out_mid[i,:,:] = data_out_mid1
            
            data_in_lr[i,:,:] = data_in_lr1
            data_out_lr[i,:,:] = data_out_lr1

        return data_in_lr, data_in_mid, data_in, data_out_lr, data_out_mid, data_out

    #######################
    data_in_lr = np.zeros((index_idx,8,16,11))
    data_in_mid = np.zeros((index_idx,8,32,22))
    data_in_hr = np.zeros((index_idx,8,64,44))

    data_out_lr = np.zeros((index_idx,8,16,11))
    data_out_mid = np.zeros((index_idx,8,32,22))
    data_out_hr = np.zeros((index_idx,8,64,44))

    for i in range(8):
        data_in_lr1, data_in_mid1, data_in_hr1, data_out_lr1, data_out_mid1, data_out_hr1 = make_data(data_in_new[:,i,:], data_out_new[:,i,:])

        data_in_lr[:,i,:,:] = data_in_lr1
        data_in_mid[:,i,:,:] = data_in_mid1
        data_in_hr[:,i,:,:] = data_in_hr1

        data_out_lr[:,i,:,:] = data_out_lr1
        data_out_mid[:,i,:,:] = data_out_mid1
        data_out_hr[:,i,:,:] = data_out_hr1

    print("finished")

    # class CustomDataset(Dataset)
    class CustomDataset(Dataset):
        def __init__(self, A, B, C, D, E):
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.E = E

        def __len__(self):
            return self.A.shape[0]

        def __getitem__(self, idx):
            input_lr = self.A[idx]
            input_hr = self.B[idx]
            output_lr = self.C[idx]
            output_hr = self.D[idx]
            geo_lr = self.E[idx]
            
            return input_lr, input_hr, output_lr, output_hr, geo_lr
        
    A_reshaped = torch.tensor(data_in_lr, dtype=torch.float32)
    B_reshaped = torch.tensor(data_in_hr, dtype=torch.float32)
    C_reshaped = torch.tensor(data_out_lr, dtype=torch.float32)
    D_reshaped = torch.tensor(data_out_hr, dtype=torch.float32)
    E_reshaped = torch.tensor(geo_data_new_lr, dtype=torch.float32)

    # seed
    seed = 42
    torch.manual_seed(seed)

    # split dataset
    dataset = CustomDataset(A_reshaped, B_reshaped, C_reshaped, D_reshaped, E_reshaped)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # data loader
    batch_size = 42
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # max and min
    maxx = maxs
    minn = mins
    return train_loader, val_loader, test_loader, maxx, minn