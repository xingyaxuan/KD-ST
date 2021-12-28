import torch
import torch.nn as nn
from data import *
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from sklearn.metrics import r2_score
import numpy.linalg as la
import csv
torch.manual_seed(2)


class CNN(nn .Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.lstm = nn.LSTM(14, 47, 1, dropout=0.2, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(7*47, 14),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.lstm (x)
        x = x[0].contiguous().view(-1,7*47*1)
        x = self.fc1(x)
        return x




def evaluate(data, X, Y,stu_net, loss_func,batch_size):
    stu_net.eval()
    yyy = np.zeros(shape=(batch_size, 14))
    rrr = np.zeros(shape=(batch_size, 14))

    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.squeeze(X)
        with torch.no_grad():
            output = stu_net(X)
        output = torch.squeeze(output)
        output = data.FNormalizeMult(output)
        Y = torch.squeeze(Y)
        Y = data.FNormalizeMult(Y)
        yyy = np.append(yyy, output, axis=0)
        rrr = np.append(rrr, Y, axis=0)
        total_loss += loss_func(output, Y).item()
        n_samples += (output.size(0) * data.m)

    yyy = yyy[batch_size:, :]
    rrr = rrr[batch_size:, :]

    mse=total_loss/n_samples
    mse = mean_squared_error(yyy[:, 0], rrr[:, 0])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yyy[:, 0], rrr[:, 0])

    return mse, rmse, mae,rrr,yyy

def train (epoch,data, X, Y, stu_net, loss1,loss2,batch_size):

    stu_net.train()
    iter=0
    total_loss =0
    n_samples =0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        lr = 0.001
        stu_net.zero_grad()
        optimizer = torch.optim.Adam(stu_net.parameters(), lr=lr)
        optimizer.zero_grad()
        X = torch.squeeze(X)
        output=stu_net(X)
        output = torch.squeeze(output)
        Y=torch.squeeze(Y)
        lr = loss2(output, Y)
        lr.backward()
        total_loss += lr.item()
        n_samples += (output.size(0) * data.m)
        grad_norm = optimizer.step()


        if iter % 5 == 0:
            print('loss',lr.item())
        iter += 1


    return total_loss / n_samples


stu_net=CNN()
stu_net.load_state_dict(torch.load("model/lstm_stu_net.pth"),strict=False)

batch_size=16
Data = DataLoaderS('../data/kd_data_1001.npy', 0.8, 0.1, 'cpu', 1,7,2)
loss1=torch.nn.SmoothL1Loss(size_average=False)
loss2=torch.nn.MSELoss(size_average=False)
best_val=100000
epoch=50
loss_loss=[]
for epoch in range(1, epoch + 1):
    epoch_start_time = time.time()
    train_loss = train(epoch,Data, Data.train[0], Data.train[1], stu_net,  loss1, loss2, batch_size)
    loss_loss.append(train_loss)
    print(time.time()-epoch_start_time,train_loss)
    mse, rmse, val_mae, rrr, yyy = evaluate(Data, Data.valid[0], Data.valid[1], stu_net, loss2, batch_size)

    if val_mae < best_val:
        with open('model/trans_stu_model.pt', 'wb') as f:
            torch.save(stu_net, f)
        best_val = val_mae
with open('model/trans_stu_model.pt', 'rb') as f:
    stu_net = torch.load(f)

'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(loss_loss, label='loss')
plt.show()
'''

val_mse, val_rmse, val_mae, val_rrr, val_yyy = evaluate(Data, Data.valid[0], Data.valid[1], stu_net, loss2, batch_size)
tes_mse, tes_rmse, tes_mae, tes_rrr, tes_yyy = evaluate(Data, Data.test[0], Data.test[1], stu_net, loss2, batch_size)
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

csvfile = open("../compare_data_save/transfer_lstm_1001.csv", 'wt', encoding="UTF8", newline='')  #
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
'''
csvfile = open("../kd_data_save/loss_trans_lstm.csv", 'wt', encoding="UTF8", newline='')  #
writer = csv.writer(csvfile, delimiter=",")
header = ['loss']
csvrow1 = []
for i in range(len(loss_loss)):
    csvrow1.append(loss_loss[i])
writer.writerow(header)
writer.writerows(zip(csvrow1))
csvfile.close()
'''