from TrainModel import load_data 
from alexnet import AlexNet
import numpy as np
import torch
import torch.nn as nn

model = AlexNet(num_classes=1000)  
model.load_state_dict(torch.load('trained_model.pth')) 
model.eval() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data_dir = 'greyScaleMini/train'
train_data, train_labels = load_data(train_data_dir)
def calculate_accuracy(model, data, labels):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    data = torch.unsqueeze(data, dim=1)
    data = torch.squeeze(data, dim=-1)
    
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
    return accuracy

train_accuracy = calculate_accuracy(model, train_data, train_labels)
print(f'Trian Accuracy: {train_accuracy}')


val_data_dir = 'greyScaleMini/val'
val_data, val_labels = load_data(val_data_dir)

val_accuracy = calculate_accuracy(model, val_data, val_labels)
print(f'Validate Accuracy: {val_accuracy}')