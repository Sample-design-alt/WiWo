import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import preprocessing

def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    if (staircase):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
    else:
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

    return optimizer

def compute_F1_score(trueY,predY):

    oriF1 = f1_score(trueY.cpu().data.numpy(),predY.cpu().data.numpy(),average='macro')
    precision = precision_score(trueY.cpu().data.numpy(),predY.cpu().data.numpy(),average='macro')
    recall = recall_score(trueY.cpu().data.numpy(),predY.cpu().data.numpy(),average='macro')

    return oriF1,precision,recall



class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

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
        if len(self.x_data.shape)==2:     #  单元时间序列要增加一个维度
            self.x_data = torch.unsqueeze(self.x_data,1)
        self.x_data = self.x_data.transpose(2,1)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))

def save_result(file,loss,accuracy,f1_score,precision,recall):
    print('accuracy:',accuracy)
    with open(file, 'a+') as f:
        f.write('{0},{1},{2},{3},{4}\n'.format(str(loss.item()),str(accuracy),f1_score,precision,recall))


