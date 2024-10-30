'''
FSCNN model with pytorch

Reference:
A. Mohammad-Alikhani, B. Nahid-Mobarakeh, M. Hsieh, "Diagnosis of Mechanical and Electrical Faults in Electric Machines Using a Lightweight Frequency-Scaled Convolutional Neural Network," IEEE Transactions on Energy Conversion, 2024.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F





class FSKenrel(nn.Module):
    def __init__(self, n_f, l_f, w=0.5, B=2, device= torch.device("cuda:0")):
        super().__init__()
        self.n_f = n_f
        self.l_f = l_f
        self.device = device
        self.w = w
        self.s = nn.Parameter(torch.tensor([i/n_f for i in range(1,n_f+1)], device=device, dtype=torch.float), requires_grad=True).to(device)
        self.B = B
        
    def forward(self):
        x = torch.linspace(-2 * torch.pi, 2 * torch.pi, self.l_f).to(self.device)
        x = x.unsqueeze(0)
        self.s.data = torch.clamp(self.s.data, min=0.01, max=16)
        s = self.s.unsqueeze(1)

        output = torch.cos(self.w * 2 * torch.pi * x * s)

        output *= torch.exp(-((x * s/self.B) ** 2) ) / (self.B * (torch.pi) ** (0.5))

        return output.unsqueeze(1)

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class PC2DLinear(nn.Module):
    def __init__(self, out_channels, input_features=6, device = torch.device("cuda:0")):
        super().__init__()
        self.device = device
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(input_features, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True).to(self.device)

    def forward(self, x):
        if self.weight.size(0) != x.size(3):
          input_features = x.size(3)
          self.weight = nn.Parameter(torch.zeros(input_features, self.out_channels, device=self.device))
          nn.init.xavier_uniform_(self.weight)
        output = torch.einsum('bijn,nm->bm', x, self.weight.double()) + self.bias
        return output.to(self.device)


class Net(nn.Module):
    def __init__(self, n_class, case_study='CWRU'):
        super().__init__()
        self.name = 'WFCNN'
        if case_study=='ITSC':
            self.l_f=50
            stride_len=2
            self.n_f = 6
            fc_input_features = 6
            channel_h = 8            
        else:
            self.l_f=25
            stride_len=2
            self.n_f = 6
            fc_input_features = 14
            channel_h = 16
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.FSKenrel_layer = FSKenrel(self.n_f, self.l_f, w=2, B=0.5, device=self.device)
            
        
        
        self.l0 = nn.BatchNorm1d(self.n_f)
        self.l1 = ConvBlock1d(self.n_f, channel_h, kernel_size=5, stride=2)
        self.pool = nn.MaxPool1d(3, stride = 2)
        
        self.l2 = ConvBlock1d(channel_h, channel_h, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(5, stride = 4, padding=2)
        
        self.l3 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,stride_len))

        self.flatten = nn.Flatten()
        self.pcl= PC2DLinear(n_class, input_features=fc_input_features, device= self.device)
 
    def forward(self, x):
        x = x.to(self.device).type(torch.cuda.DoubleTensor)
        FSKenrel_layer = self.FSKenrel_layer()
        fc = F.conv1d(x, FSKenrel_layer.to(self.device).type(torch.cuda.DoubleTensor),stride=8)
        f0 = self.l0(fc)

        f1 = self.l1(f0).type(torch.cuda.DoubleTensor)
        f1 = F.relu(f1)
        f1 = self.pool(f1)

        f3 = self.l2(f1)
        f3 = F.relu(f3)
        f3 = self.pool2(f3)

        f4 = f3.unsqueeze(1)
        f5 = self.l3(f4)
        f5 = F.relu(f5)

        f6 = f5.transpose(2,3)

        out = self.pcl(f6)
        
        return out
