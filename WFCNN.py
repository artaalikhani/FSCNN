'''
WFCNN model with pytorch

Reference:

'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ConvBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

def morlet(M, w=4.0, s=0.1,B=4.0, complete=False):
    x = torch.linspace(-2*torch.pi,2*torch.pi,M)
    output = torch.cos(w*2*torch.pi*x/s)#np.exp(1j*w*2*np.pi*x)

    if complete:
        x -= torch.exp(-0.5*(w**2))

    output *= torch.exp(-((x/s)**2)/B) * (torch.pi*B)**(-0.5)

    return output#.float()

def gaus(M, complete=False):
    x = torch.linspace(-2*torch.pi,2*torch.pi,M)
    output = 1#torch.cos(w*2*torch.pi*x/s)#np.exp(1j*w*2*np.pi*x)


    output *= torch.exp(-((x/s)**2)) * (torch.pi)**(-0.5)

    return output#.float()

class Net(nn.Module):
    def __init__(self, n_class, case_study='CWRU'):
        super(Net, self).__init__()
        self.name = 'WFCNN'
        if case_study=='CWRU':
            fsize=25
            stride_len=2
        else:
            fsize=50
            stride_len=1
        n_f = 16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.filter = torch.zeros(n_f,1,fsize)
        
        for j in range(n_f):
            self.filter[j,0,:] = morlet(fsize, w=4.0, s=j*0.5+0.1,B=4, complete=False)
            
        
        channel_h = 16
        self.l0 = nn.BatchNorm1d(n_f)
        self.l1 = ConvBlock1d(n_f, channel_h, kernel_size=5, stride=2)
        self.pool = nn.MaxPool1d(3, stride = 2)
        
        self.l2 = ConvBlock1d(channel_h, channel_h, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(5, stride = 4, padding=2)
        
        self.l3 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,stride_len))

        self.flatten = nn.Flatten()
        self.fc= nn.LazyLinear(n_class)
        
 
    def forward(self, x):
        x = x.to(self.device).type(torch.cuda.DoubleTensor)
        fc = F.conv1d(x, self.filter.to(self.device).type(torch.cuda.DoubleTensor),stride=8)
        f0 = self.l0(fc)
        f1 = self.l1(f0)
        f1 = F.relu(f1)
        f2 = self.pool(f1)
        f3 = self.l2(f2)
        f3 = F.relu(f3)
        f3 = self.pool2(f3)
        f4 = f3[:,None,:,:]
        f5 = self.l3(f4)
        f5 = F.relu(f5)
        f6 = self.flatten(f5)
        out=self.fc(f6)
        return out