#!/usr/bin/env python

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import time



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.feature_sizes = (8,16,32,64,128,256)
        f1,f2,f3,f4,f5,f6 = self.feature_sizes



        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(2, f1, 5, stride=2)
        self.conv2 = nn.Conv2d(f1, f2, 5, stride=2)
        self.conv3 = nn.Conv2d(f2, f3, 5, stride=2)
        self.conv4 = nn.Conv2d(f3, f4, 5, stride=2)
        self.conv5 = nn.Conv2d(f4, f5, 5, stride=2)

        self.fc1_size = f5*1*1
        self.fc1 = nn.Linear(self.fc1_size,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,8)
        self.fc4 = nn.Linear(8,2)

        torch.nn.init.normal(self.conv1.weight, std=0.00001)
        torch.nn.init.normal(self.conv2.weight, std=0.00001)
        torch.nn.init.normal(self.conv3.weight, std=0.00001)
        torch.nn.init.normal(self.conv4.weight, std=0.00001)
        torch.nn.init.normal(self.conv5.weight, std=0.00001)
        torch.nn.init.normal(self.conv6.weight, std=0.00001)

        torch.nn.init.normal(self.fc1.weight, std=0.00001)
        torch.nn.init.normal(self.fc2.weight, std=0.00001)
        torch.nn.init.normal(self.fc3.weight, std=0.00001)
        torch.nn.init.normal(self.fc4.weight, std=0.00001)
    




    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(self.fc1_size)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.tanh(self.fc4(x)) + 0.5
        return x






if __name__ == "__main__":
    print("create net")
    net = Net()
    net.cuda()
    test_img = torch.from_numpy(np.random.rand(1,2,128,128))
    input_var = torch.autograd.Variable(test_img).float().cuda()


    print("start")
    n = 1000
    out=net.forward(input_var)
    start_time = time.time()
    for i in range(n):
        out=net.forward(input_var)
    run_time = (time.time() - start_time)/float(n)
    print("%ims, %ihz" % (run_time*1000,1/run_time))
    print(net)
    count = 0
    for x in net.parameters():
        val = 1
        for dim in x.shape:
            val *= dim
        count += val
    print("%i parameters" % count)
    print("input",input_var.shape)
    print("output",out.shape)
