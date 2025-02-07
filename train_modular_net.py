# -*- coding: utf-8 -*-
"""train_modular_net.ipynb

By DK
"""

import os
print(os.getcwd())

from  periodictable import elements
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
#from pymatgen.core.composition import Composition

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {dev} device")

def is_nan(x):
    return (x != x)



# @title Default title text
# model constuction
## model constuction
def create_model():

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()

#BackBone
           self.seq1 = nn.Sequential(
               nn.Conv2d(1, 32, 7, stride=1, padding=(3,3)),
               nn.ReLU(),
               nn.Conv2d(32, 32, 5, stride=1, padding=(2,2)),
               nn.ReLU()
               )

#Classification branch
           self.seq2 = nn.Sequential(
               nn.Conv2d(32, 64, 3, stride=2),
               nn.ReLU(),
               nn.MaxPool2d(3, stride=2),
               nn.Dropout(0.25),
               nn.Flatten(),
               nn.Linear(128, 512),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(512, 2),
               nn.Softmax(dim=1)
           )

#Prediction Branch
           self.seq3 = nn.Sequential(
               nn.Conv2d(32, 64, 3, stride=2),
               nn.ReLU(),
               nn.Conv2d(64, 64, 3, stride=2),
               nn.ReLU(),
               nn.Conv2d(64, 128, 1, stride=1),
               nn.ReLU(),
               nn.Conv2d(128, 64, 1, stride=1),
               nn.ReLU(),
               nn.Dropout(0.25),
               nn.Flatten(),
               nn.Linear(128, 512),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(512, 1)
           )

       def forward(self, x):
           x = self.seq1(x)
           output1 = self.seq2(x)
           output2 = self.seq3(x)
           return [output2, output1]
   model = Net()
   print(model)
   return model
#



#train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2])
train_ds=torch.load('train_ds.pt')
val_ds=torch.load('val_ds.pt')
# 16414*0.2=3282.8 /testing/
# 16414-3282=13132  /training/ page 7 manuscript

#print(f"Shape of train_ds: {train_ds.shape}")
#print(f"Shape of val_ds  : {val_ds.shape}")

total_tc_train = 0
for i in range(len(train_ds)):
    total_tc_train = total_tc_train + (train_ds[i][1][2])

print("--------------trantotal_tc_traine_tc---------------")
print(len(train_ds), total_tc_train/len(train_ds))
#print(f"Tensor: \n {train_ds} \n")

total_tc_val = 0
for i in range(len(val_ds)):
    total_tc_val = total_tc_val + (val_ds[i][1][2])

#for i in range(1, 10, 1):
#    print(val_ds[i][1][2], val_ds[i][1][1])

print("---------------total_tc_val-------------------")
print(len(val_ds), total_tc_val/len(val_ds))

#quit()

net = create_model()
net.to(dev)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss(reduction='sum')

epochs = 5000

train_loader = DataLoader(train_ds, 64, drop_last=True, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, 1, drop_last=True, shuffle=True, num_workers=0)

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

pred_absdiff = []
train_accuracy = []
train_loss = []
net.double()

# prediction branch training
for epoch in range(epochs):
    if epoch == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
#(1 item) [{'params': [Parameter with shape torch.Size([32, 1, 7, 7]), Parameter with shape torch.Size([32]), Parameter with shape torch.Size([32, 32, 5, 5]), Parameter with shape torch.Size([32]), Parameter with shape torch.Size([64, 32, 3, 3]), ...], 'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, ...}]
#(variable) param_group: Dict[str, Any]

    net.train()
    #print("Epoch=", epoch)


    for xb, yb in train_loader:

        def closure():
            optimizer.zero_grad()
            xb_g = xb.to(dev)
            #udo
            #print(xb_g)
            #m = nn.Flatten()
            #output = m(xb_g)
            #print('size is ?')
            #print(output.size())
            #print(output)

            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc
            pred = net(xb_g)
            loss = loss_fn(pred[0].flatten(), yb_g_split_1.flatten())

           # print(pred[0])
           # print( "-----1---")
           # print(pred[1])
           # wait = input("Press Enter to continue.")


            #wait = input("Press Enter to continue.")
            #print(pred[0].flatten())
            #print( "-----1---")
            #print(pred[1].flatten())

           # wait = input("Press Enter to continue.")

            loss.backward()
            return loss

        optimizer.step(closure)
    net.eval()
    valid_loss = 0
    diff = 0
    count = 0
    with torch.no_grad():
        for xb, yb in train_loader:

            xb_g = xb.to(dev)

            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc
            pred = net(xb_g)
            valid_loss = valid_loss + loss_fn(pred[0].flatten(), yb_g_split_1.flatten())

            pred_data = pred[0].to(torch.device("cpu"))
            pred_data = pred_data.numpy().flatten()
            gt_data = yb_g_split[1].numpy().flatten()


            for i in range(len(pred_data)):
               diff = diff + np.absolute(pred_data[i] - gt_data[i])

            class_data = pred[1].to(torch.device("cpu"))
            gt_data = yb_g_split[0].numpy()
            for i in range(len(class_data)):
                pred_result = 1
                if class_data[i][0] > class_data[i][1]:
                    pred_result = 0
                gt_data_result = 0
                if gt_data[i][0] < gt_data[i][1]:
                    gt_data_result = 1
                if pred_result == gt_data_result:
                    count+=1


    pred_absdiff.append((diff/(len(train_loader)*64)))
    train_accuracy.append((count/(len(train_loader)*64)))

    temp = valid_loss.to(torch.device("cpu"))
    train_loss.append((temp/(len(train_loader)*64)))
    print(epoch, (temp/(len(train_loader)*64)), (diff/(len(train_loader)*64)), (count/(len(train_loader)*64)))

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),pred_absdiff)
plt.title('Accuray')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['Train Pred Acc'],loc = 'lower right')
plt.savefig("./pred_train_accuracy.png",dpi = 600)

np.savetxt("pred_train_accu.csv", pred_absdiff, delimiter=",", fmt='%f')

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),train_loss)
plt.title('MSE Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['Train MSE Loss'],loc = 'lower right')
plt.savefig("./pred_train_loss.png",dpi = 600)

np.savetxt("pred_train_loss.csv", train_loss, delimiter=",", fmt='%f')

for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0001

net.train()

# freeze main + prediction branch
for param in net.seq1.parameters():
    param.requires_grad = False

for param in net.seq3.parameters():
    param.requires_grad = False

epochs = 5000
pred_absdiff = []
train_accuracy = []
train_loss = []

# train classification branch
for epoch in range(epochs):
    if epoch == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    net.train()

    for xb, yb in train_loader:

        def closure():
            optimizer.zero_grad()
            xb_g = xb.to(dev)
            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc
            pred = net(xb_g)
            loss = loss_fn(pred[1], yb_g_split_0)
            loss.backward()
            return loss

        optimizer.step(closure)

    net.eval()
    valid_loss = 0
    diff = 0
    count = 0
    with torch.no_grad():
        for xb, yb in train_loader:

            xb_g = xb.to(dev)
            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc
            pred = net(xb_g)
            valid_loss = valid_loss + loss_fn(pred[1], yb_g_split_0)



            pred_data = pred[0].to(torch.device("cpu"))
            pred_data = pred_data.numpy().flatten()
            gt_data_tc = yb_g_split[1].numpy().flatten()

            class_data = pred[1].to(torch.device("cpu"))
            gt_data = yb_g_split[0].numpy()
            for i in range(len(class_data)):
                pred_result = 1
                if class_data[i][0] > class_data[i][1]:
                    pred_result = 0
                gt_data_result = 0
                if gt_data[i][0] < gt_data[i][1]:
                    gt_data_result = 1
                if pred_result == gt_data_result:
                    count+=1

                if pred_result == gt_data_result:
                    if pred_result == 1:
                        diff = diff + np.absolute(pred_data[i] - gt_data_tc[i])




    pred_absdiff.append((diff/(len(train_loader)*64)))
    train_accuracy.append((count/(len(train_loader)*64)))

    temp = valid_loss.to(torch.device("cpu"))
    train_loss.append((temp/(len(train_loader)*64)))
    print(epoch, (temp/(len(train_loader)*64)), (diff/(len(train_loader)*64)), (count/(len(train_loader)*64)))

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),train_accuracy)
plt.title('Accuray')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['Train Class Acc'],loc = 'lower right')
plt.savefig("./class_accuracy.png",dpi = 600)

np.savetxt("class_train_accu.csv", train_accuracy, delimiter=",", fmt='%f')

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),train_loss)
plt.title('MSE Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['Train MSE Loss'],loc = 'lower right')
plt.savefig("./class_train_loss.png",dpi = 600)

np.savetxt("class_train_loss.csv", train_loss, delimiter=",", fmt='%f')

# full model testing
net.eval()
diff = 0
count = 0
tp=0
tn=0
fp=0
fn=0
with torch.no_grad():
    for xb, yb in val_loader:

        xb_g = xb.to(dev)
        yb_g_split = np.split(yb, [2], 1)
        yb_g_split_0 = (yb_g_split[0]).to(dev) #class
        yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc
        pred = net(xb_g)

        pred_data = pred[0].to(torch.device("cpu"))
        pred_data = pred_data.numpy().flatten()
        gt_data_tc = yb_g_split[1].numpy().flatten()

        class_data = pred[1].to(torch.device("cpu"))
        gt_data = yb_g_split[0].numpy()
        for i in range(len(class_data)):
            pred_result = 1
            if class_data[i][0] > class_data[i][1]:
                pred_result = 0
            gt_data_result = 0
            if gt_data[i][0] < gt_data[i][1]:
                gt_data_result = 1
            if pred_result == gt_data_result:
                count+=1
            if pred_result == gt_data_result and gt_data_result == 1:
                tp += 1
            if pred_result == gt_data_result and gt_data_result == 0:
                tn += 1
            if pred_result == 0 and gt_data_result == 1:
                fn += 1
            if pred_result == 1 and gt_data_result == 0:           
                fp += 1
            if pred_result == gt_data_result:
                if pred_result == 1:
                    diff = diff + np.absolute(pred_data[i] - gt_data_tc[i])

print((diff/(len(val_loader)*1)), (count/(len(val_loader)*1)))
print("Accuracy: ", (tp+tn)/(tp+tn+fp+fn))
print("Precision: ", (tp)/(tp+fp))
print("Recall: ", (tp)/(tp+fn))
print("tp tn fp fn", [tp,tn,fp,fn])
# model export to onnx
net.to(torch.device("cpu"))
net.float()
dummy_input = torch.randn(1, 1, 10, 12)
torch.onnx.export(net, dummy_input, "model_pytorch.onnx", verbose=True)
model_scripted = torch.jit.script(net) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save
