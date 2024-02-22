import os
import numpy as np
import torch
import torch.nn as nn
from alexnet import AlexNet 
from tqdm import tqdm 
learning_rate = 0.001
num_epochs = 10
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir):
    data = []
    labels = []
    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(folder_path, file)
                    image = np.load(file_path)
                    data.append(image)
                    labels.append(label)
    return np.array(data), np.array(labels)

train_data_dir = 'greyScaleMini/train'
train_data, train_labels = load_data(train_data_dir)

indices = np.arange(len(train_data))
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]

train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
train_data = torch.unsqueeze(train_data, dim=1)
train_data = torch.squeeze(train_data, dim=-1)

model = AlexNet(num_classes=1000).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='samples') as pbar:
        for i in range(0, len(train_data), batch_size):
            batch_x = train_data[i:i + batch_size]
            batch_y = train_labels[i:i + batch_size]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            pbar.update(len(batch_x))
            pbar.set_postfix({'Loss': loss.item()})
        print('Epoch %i: Loss: %f' % (epoch + 1, loss.item()))

torch.save(model.state_dict(), 'trained_model.pth')