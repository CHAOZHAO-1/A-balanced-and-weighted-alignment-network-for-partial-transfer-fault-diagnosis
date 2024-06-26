from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio
def zscore(Z):
    Zmax, Zmin = Z.max(axis=0), Z.min(axis=0)
    Z = (Z - Zmin) / (Zmax - Zmin)
    return Z


def load_training(notargetlist,root_path,dir, fft1, class_num , batch_size, kwargs):
    data = scio.loadmat(root_path)
    if fft1==True:
        train_fea=zscore((abs(fft(data[dir]))[:,0:512]).transpose()).transpose()
    if fft1==False:
        train_fea = zscore(data[dir].transpose()).transpose()

    train_label = torch.zeros((800 * class_num))
    for i in range(800 * class_num):
        train_label[i] = i // 800

    long = len(notargetlist)
    if long!=0:
        b = np.linspace(0, 799, 800).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 800 + b)
        train_fea = np.delete(train_fea, removelist, axis=0)
        train_label = np.delete(train_label, removelist, axis=0)

    print(train_fea.shape)
    train_label = train_label.long()
    train_fea=torch.from_numpy(train_fea)
    train_fea= torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader







def load_testing(notargetlist, root_path, dir, fft1, class_num, batch_size, kwargs):
    data = scio.loadmat(root_path)
    if fft1 == True:
        train_fea = zscore((abs(fft(data[dir]))[:, 0:512]).transpose()).transpose()
    if fft1 == False:
        train_fea = zscore(data[dir].transpose()).transpose()
    train_label = torch.zeros((200 * class_num))
    for i in range(200 * class_num):
        train_label[i] = i // 200
    #
    long = len(notargetlist)
    if long!=0:
        b = np.linspace(0, 199, 200).reshape(-1, 1)
        removelist = []
        for i in range(long):
            removelist = np.append(removelist, (notargetlist[i] - 1) * 200 + b)
        train_fea = np.delete(train_fea, removelist, axis=0)
        train_label = np.delete(train_label, removelist, axis=0)



    print(train_fea.shape)
    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea)
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader
