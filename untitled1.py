import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader



def get_img_paths(path):
    paths = []
    labels = []
    for label in os.listdir(path):
        img_dir = os.path.join(path, label)
        for img in os.listdir(img_dir):
            paths.append(os.path.join(img_dir, img))
            labels.append(label)

    return pd.DataFrame({'path':paths, 'label':labels})


#training data

train = get_img_paths("C:\\Users\\Shubhashish Chanda\\Desktop\\classification(proj)\\archive\\DATASET\\TRAIN")
train.head()
train.info


#testing data

test = get_img_paths("C:\\Users\\Shubhashish Chanda\\Desktop\\classification(proj)\\archive\\DATASET\\TEST")
test.head()
test.info

#Label Encoding

conversion = {'O': 0, 'R': 1}

train.label = train.label.map(conversion)
test.label = test.label.map(conversion)

train.head()



class WasteData(Dataset):
    def __init__(self, dir_lbl, transform=None):
        self.dir_lbl = dir_lbl
        self.transform = transform

    def __len__(self):
        return len(self.dir_lbl)

    def __getitem__(self, idx):
        img_dir_lbl = self.dir_lbl.iloc[idx]
        img_dir = img_dir_lbl.path
        label = img_dir_lbl.label
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label



data_transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225])
])

train_dataset = WasteData(train, data_transform)
train_dataset

train_size = int(0.9 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

test_dataset = WasteData(test, data_transform)
test_dataset


batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


for i in range(5):
    train_features, train_labels = next(iter(train_dataloader))
    img = train_features[0].squeeze().numpy().transpose((1, 2, 0))

    label = train_labels[0]
    print(f"Label {i+1}: {label}")
    plt.imshow(img)
    plt.show()
    
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

torch.cuda.is_available()




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
    
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
            
        self.fc1 = nn.Linear(64*24*24, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.fc4 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    
net = Net().to(device)


import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)




train_loss = []
val_loss = []
epochs = 4
for epoch in range(epochs): 
    epoch_loss = 0.0
    epoch_loss_val = 0.0
    running_loss = 0.0
    print('Training:')
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].float().to(device)

        optimizer.zero_grad()

        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            loss = running_loss / 10
            epoch_loss += loss
            print(f'\t[{epoch + 1}, {i + 1:5d}] loss: {loss:.6f}')
            running_loss = 0.0
     
    print('Validation:')
    running_loss_valid = 0.0
    for i, data in enumerate(valid_dataloader, 0):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].float().to(device)

            outputs = net(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss_valid += loss.item()
            if i % 10 == 9:
                loss = running_loss_valid / 10
                epoch_loss_val += loss
                print(f'\t[{epoch + 1}, {i + 1:5d}] loss: {loss:.6f}')
                running_loss_valid = 0.0
                
    train_loss.append(epoch_loss)
    val_loss.append(epoch_loss_val)

print('Finished Training and Validation')



plt.figure(figsize=(20,6));
sns.lineplot(x=list(range(epochs)), y=train_loss)
sns.lineplot(x=list(range(epochs)), y=val_loss)
plt.legend(['Training loss', 'Validation loss']) 
