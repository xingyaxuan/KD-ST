import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import time
import math
import csv

from GAN import sagan_models
#from sklearn import decomposition
from sklearn.metrics import r2_score
import numpy.linalg as la
from thop import profile
torch.manual_seed(2)

def gan_backward(loss1,output,output1):
    features_k=14
    gan = sagan_models.Discriminator(2, features_k)
    gan_optim = torch.optim.Adam(gan.parameters(), lr=0.001, weight_decay=0.00001)
    gan_optim.zero_grad()
    gan_stu = gan(output)
    gan_tea = gan(output1)
    gan_loss=loss1(gan_stu,gan_tea)
    gan_loss.backward(retain_graph=True)
    GAN_loss=gan_loss.item()
    gan_optim.step()
    return GAN_loss


class CNN(nn .Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(14,50, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(7*50*1, 14),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.contiguous().view(-1,7*50*1)
        x = self.fc1(x)
        return x




def evaluate(data, X, Y,stu_net, loss_func,batch_size):
    stu_net.eval()
    yyy = np.zeros(shape=(batch_size, 14))
    rrr = np.zeros(shape=(batch_size, 14))

    total_loss = 0
    n_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.squeeze(X)
        X = X.permute(0, 2, 1)
        with torch.no_grad():
            output = stu_net(X)

        output = data.FNormalizeMult(output)
        Y = torch.squeeze(Y)
        Y = data.FNormalizeMult(Y)
        yyy = np.append(yyy, output.detach().numpy(), axis=0)
        rrr = np.append(rrr, Y.detach().numpy(), axis=0)
        total_loss += loss_func(output, Y).item()
        n_samples += (output.size(0) * data.m)

    yyy = yyy[batch_size:, :]
    rrr = rrr[batch_size:, :]

    mse = mean_squared_error(yyy[:, 0], rrr[:, 0])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yyy[:, 0], rrr[:, 0])

    return mse, rmse, mae,rrr,yyy

def train (epoch,data, X, Y, stu_net, teacher_model,loss1,loss2,batch_size):

    stu_net.train()
    iter=0
    total_loss =0
    n_samples =0
    timea=0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        lr = 0.001
        if epoch >=5:
            lr=lr*0.1
        X_tea = X.transpose(1, 3)

        #Please design the teacher network according to your own needs
        with open('model/model.pt', 'rb') as f:
            teacher_model = torch.load(f)
        with torch.no_grad():
            output1 = teacher_model(X_tea)
        output1 = torch.squeeze(output1)
        test_start_time = time.time()
        stu_net.zero_grad()
        optimizer = torch.optim.Adam(stu_net.parameters(), lr=lr)
        optimizer.zero_grad()
        X = torch.squeeze(X)
        X = X.permute(0, 2, 1)
        output=stu_net(X)
        Y = torch.squeeze(Y)
        GAN_loss = gan_backward(loss1, output, output1)
        #'''#key code!

        dist = np.zeros((output.shape[0],output.shape[1]))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                dist[i, j] = (Y[i, j] -output1[i, j]) ** 2
        print(dist.max())
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if dist[i, j] >= 0.3:
                    Y[i,j]=output[i, j]
        #'''
        lr =0.6* loss2(output, Y)+0.4*gan_backward(loss1, output, output1)
        lr.backward()
        total_loss += lr.item()
        n_samples += (output.size(0) * data.m)
        grad_norm = optimizer.step()
        test_end_time = time.time() - test_start_time
        timea += test_end_time
        if iter % 5 == 0:
            print('loss',lr.item())
        iter += 1


    return total_loss / n_samples, optimizer,timea




batch_size=16
Data = DataLoaderS('../data/kd_data_1001.npy', 0.8, 0.1, 'cpu', 1,7,2)
loss_func = torch.nn.MSELoss(size_average=False)
loss1=torch.nn.L1Loss(size_average=False)
loss2=torch.nn.MSELoss(size_average=False)
stu_net= CNN()
'''
inputs=torch.randn(16, 14, 7)
total_ops, total_params = profile(stu_net, (inputs,), verbose=False)
print(total_ops,total_params)
'''

best_val=100000
epoch=5
for epoch in range(1, epoch + 1):
    epoch_start_time = time.time()
    train_loss,optim,timea = train(epoch,Data, Data.train[0], Data.train[1], stu_net, teacher_model, loss1, loss2, batch_size)
    print(time.time()-epoch_start_time,train_loss,timea)
    mse, rmse, val_mae, rrr, yyy = evaluate(Data, Data.valid[0], Data.valid[1], stu_net, loss_func, batch_size)
    all_states= {"net": stu_net.state_dict(), "Adam": optim.state_dict(), "epoch": epoch}
    if val_mae < best_val:
        with open('model/1cnn_stu_model.pt', 'wb') as f:
            torch.save(stu_net, f)
            torch.save(obj=stu_net.state_dict(), f="model/1cnn_stu_net.pth")
        best_val = val_mae
with open('model/1cnn_stu_model.pt', 'rb') as f:
    stu_net = torch.load(f)
'''
nParams = sum([p.nelement() for p in stu_net.parameters()])
print('Number of model parameters is', nParams, flush=True)
'''

val_mse, val_rmse, val_mae, val_rrr, val_yyy = evaluate(Data, Data.valid[0], Data.valid[1], stu_net, loss_func, batch_size)
tes_mse, tes_rmse, tes_mae, tes_rrr, tes_yyy = evaluate(Data, Data.test[0], Data.test[1], stu_net, loss_func, batch_size)
print('tes_yyy',tes_yyy.shape)
print('tes_rrr',tes_rrr.shape)
r2 = r2_score(tes_yyy, tes_rrr)
F_norm = la.norm(tes_rrr- tes_yyy, 'fro') / la.norm(tes_rrr, 'fro')
var = 1 - (np.var(tes_rrr - tes_yyy)) / np.var(tes_rrr)
mse_all = []
rmse_all = []
mae_all = []
N_node = 14
for i in range(N_node):
    mse = mean_squared_error(tes_yyy[:, i], tes_rrr[:, i])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(tes_yyy[:, i],tes_rrr[:, i])
    mse_all.append(mse)
    rmse_all.append(rmse)
    mae_all.append(mae)
print(mae_all)

csvfile = open("../compare_data_save/1cnn_1001_kd_mms-st.csv", 'wt', encoding="UTF8", newline='')  #
writer = csv.writer(csvfile, delimiter=",")
header = ['kd_mms-st_mse', 'kd_mms-st_rmse', 'kd_mms-st_mae', 'kd_mms-st_acc', 'kd_mms-st_r2', 'kd_mms-st_var']
csvrow1 = []
csvrow2 = []
csvrow3 = []
csvrow4 = []
csvrow5 = []
csvrow6 = []

for i in range(len(mse_all)):
    csvrow1.append(mse_all[i])
    csvrow2.append(rmse_all[i])
    csvrow3.append(mae_all[i])
    csvrow4.append(F_norm)
    csvrow5.append(r2)
    csvrow6.append(var)
writer.writerow(header)
writer.writerows(zip(csvrow1, csvrow2, csvrow3, csvrow4, csvrow5, csvrow6))
csvfile.close()

csvfile = open("../compare_data_save/1cnn_1001_test_data.csv", 'wt', encoding="UTF8", newline='')  #
writer = csv.writer(csvfile, delimiter=",")
header = ['test_y4', 'predicted_y4', 'test_y7', 'predicted_y7', 'test_y11', 'predicted_y11']
csvrow1 = []
csvrow2 = []
csvrow3 = []
csvrow4 = []
csvrow5 = []
csvrow6 = []
for i in range(160):
    csvrow1.append(tes_rrr[i, 4])
    csvrow2.append(tes_yyy[i, 4])
    csvrow3.append(tes_rrr[i, 7])
    csvrow4.append(tes_yyy[i, 7])
    csvrow5.append(tes_rrr[i, 11])
    csvrow6.append(tes_yyy[i, 11])
writer.writerow(header)
writer.writerows(zip(csvrow1, csvrow2, csvrow3, csvrow4, csvrow5, csvrow6))

csvfile.close()
