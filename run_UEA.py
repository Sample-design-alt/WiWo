import torch
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from torch.autograd import Variable
from t2swin_transformer import SwinTransformer
from utils import compute_F1_score,exponential_decay,save_result,DealDataset
from read_UEA import load_UCR
import glob
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def GetDataAndNet(wa,prob,mask):
    train_loader,test_loader,num_class = load_UCR(archive)


    time_stmp = train_loader._get_iterator().next()[0].shape[1]
    in_channel = train_loader._get_iterator().next()[0].shape[2]



    net = SwinTransformer(
        t=time_stmp,
        down_dim=8192,
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=in_channel,
        num_classes=num_class,
        head_dim=32,
        window_size=64,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True,
        wa=wa,
        prob=prob,
        mask=mask,
    ).to(device)
    return train_loader,test_loader,net,num_class


def test():
    correct = 0
    total_pred = torch.tensor([], dtype=torch.int64).to(device)
    total_true = torch.tensor([], dtype=torch.int64).to(device)

    for batch_id, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        net.eval()
        x = Variable(x).float().to(device)
        y = Variable(y).to(device)

        pred_y = net(x)
        _, y_pred = torch.max(pred_y, -1)
        correct += (y_pred.cpu()==y.cpu()).sum().item()
        total_pred = torch.cat([total_pred, y_pred], dim=0)
        total_true = torch.cat([total_true, y], dim=0)


    f1_score,precision,recall = compute_F1_score(total_true,total_pred)
    return correct,f1_score,precision,recall




def train(optimizer,wa,prob,mask):
    for epoch in range(n_epochs):
        file = r'./result/result_{0}_{1}_{2}_{3}.csv'.format(str(wa),str(prob),str(mask),archive)
        ls=[]
        start_time,stop_time = 0,0
        for batch_id,(x,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            net.train()
            optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1,0.90)

            x = Variable(x).float().to(device)
            y = Variable(y).to(device)
            start_time = time.time()
            pred_y = net(x)
            stop_time = time.time()
            #loss
            loss = loss_func(pred_y,y.to(torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, y_pred = torch.max(pred_y, -1)
            ls.append(loss)
        if epoch >150:
            correct,f1_score,precision,recall = test()
            print(stop_time-start_time)
            save_result(file,ls[-1],correct/test_loader.dataset.__len__(),f1_score,precision,recall)



if __name__ == '__main__':
        for wa in range(1,2):  #  wa/prob/mask---->bottom
            for prob in range(1,2):
                for mask in range(1,2):

                        archives = glob.glob(r'data/*')
                        for archive_path in archives:
                            archive = os.path.split(archive_path)[-1]
                            print(archive)
                            file = r'./result/result_{0}_{1}_{2}_{3}.csv'.format(str(wa), str(prob), str(mask),
                                                                                 archive)
                            if os.path.exists(file):
                                print('run over')
                                continue
                            try:
                                train_loader,test_loader,net,num_class=GetDataAndNet(wa,prob,mask)

                                LEARNING_RATE = 0.001
                                optimizer = torch.optim.Adam(
                                    net.parameters(),
                                    lr=LEARNING_RATE,
                                    betas=(0.9, 0.999),
                                    eps=1e-08
                                )
                                global_epoch = 0
                                global_step = 0
                                best_tst_accuracy = 0.0
                                COMPUTE_TRN_METRICS = True
                                n_epochs = 200

                                loss_func = torch.nn.CrossEntropyLoss()

                                train(optimizer,wa,prob,mask)
                            except:
                                file = r'./result/result_{0}_{1}_{2}_{3}.csv'.format(str(wa), str(prob), str(mask),
                                                                                     archive)
                                with open(file, 'a+') as f:
                                    f.write('error\n')
                                continue