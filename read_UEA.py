import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from utils import DealDataset
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DealDataset(Dataset):


    def __init__(self,filename):
        data = np.loadtxt(filename, delimiter='\t')
        Y = data[:, 0]
        X = data[:, 1:]

        Y = preprocessing.LabelEncoder().fit(Y).transform(Y)
        X[np.isnan(X)] = 0
        X = preprocessing.scale(X)
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(Y)
        self.len = X.shape[0]
        if len(self.x_data.shape)==2:
            self.x_data = torch.unsqueeze(self.x_data,1)
        self.x_data = self.x_data.transpose(2,1)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))




def load_UCR(archive_name):

    root_dir = r'data'

    train_path=os.path.join(os.path.join(root_dir,archive_name),archive_name+'_TRAIN.tsv')
    test_path=os.path.join(os.path.join(root_dir,archive_name),archive_name+'_TEST.tsv')

    TrainDataset = DealDataset(train_path)
    TestDataset = DealDataset(test_path)

    train_loader = DataLoader(dataset=TrainDataset,
                               batch_size=16,
                               shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                               batch_size=16,
                               shuffle=True)



    num_class = DealDataset(train_path).num_class()



    return train_loader,test_loader,num_class



if __name__ == '__main__':
    load_UCR('Ering')
