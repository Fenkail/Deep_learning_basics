import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1)
        self.conv2 = nn.Conv2d(20, 30, 3, stride=1)
        self.conv3 = nn.Conv2d(30, 20, 3, stride=1)
        self.conv4 = nn.Conv2d(20, 10, 3, stride=1)
        self.pool1 = nn.AvgPool2d(2, padding=0, stride=2)
        self.pool2 = nn.AvgPool2d(2, padding=0, stride=2)
        self.pool3 = nn.AvgPool2d(2, padding=0, stride=2)
        self.pool4 = nn.AvgPool2d(2, padding=0, stride=2)
        self.liner = nn.Linear(23*23*10,10)
        self.relu = nn.ReLU()

    
    def forward_emb(self, input):
        out = self.conv1(input)
        out = self.pool1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.pool4(out)
        out = self.relu(out)
        out = out.view(23*23*10)
        out = self.liner(out)
        return out

if __name__ == "__main__":
    model = ModelCNN()
    input = torch.rand(1, 1, 400, 400)
    out = model.forward_emb(input)

