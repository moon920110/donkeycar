'''

pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion. 

'''


import os
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import donkeycar as dk
from torch.optim import Adam, SGD
from tqdm import trange, tqdm


class TorchPilot(object):
    '''
    Base class for Torch models that will provide steering and throttle to guide a car.
    '''
    def __init__(self):
        self.model = None
        self.optimizer = None
 
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        
    def shutdown(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        if optimizer_type == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=rate, weight_decay=decay)
        elif optimizer_type == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=rate, weight_decay=decay)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)
    
    def train(self, train_loader, val_loader, model_name, epochs=100, verbose=1):
        model_path = os.path.expanduser(model_name)
        criterion = nn.MSELoss()
        min_val_loss = 9999
        no_improve_cnt = 0

        with trange(epochs) as t:
            for epoch in t:
                with tqdm(train_loader) as loader:
                    train_step = 0
                    losses = 0
                    self.model.train()
                    for data in loader:
                        train_step += 1
                        img = data['img'].float()
                        speed = data['speed'].float()
                        target = data['out']  # [steering, throttle]

                        out = self.model(img, speed)
                        loss = criterion(out, target)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        losses += loss.detach()                        

                        loader.set_postfix(train_loss=loss.detach())

                with tqdm(val_loader) as loader:
                    self.model.eval()
                    val_step = 0
                    val_losses = 0
                    for data in loader:
                        val_step += 1
                        img = data['img'].float()
                        speed = data['speed'].float()
                        target = data['out']  # [steering, throttle]

                        out = self.model(img, speed)
                        loss = criterion(out, target)

                        val_losses += loss.detach()

                        loader.set_postfix(val_loss=loss.detach())
                    val_losses /= val_step
                    
                    if val_losses < min_val_loss:
                        no_improve_cnt = 0
                        min_val_loss = val_losses
                    else:
                        no_improve_cnt += 1

                t.set_postfix(avg_loss=losses/train_step)

                if no_improve_cnt == 6:
                    print("EARLY STOP: {}".format(min_val_loss))
                    self.save(model_name)
                    break

    
class TorchIL(TorchPilot):
    '''
    Pretrain network before RL using IL
    '''
    def __init__(self, *args, **kwargs):
        super(TorchIL, self).__init__(*args, **kwargs)
        self.model = TorchDriver()

    def run(self, img_arr, speed):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        speed = np.reshape(np.array(speed), (1, 1))
        outputs = self.model(img_arr, speed)

        return outputs[0][0], outputs[0][1]


class Perception(nn.Module):
    def __init__(self):
        super(Perception, self).__init__()
        self.img_extractor = nn.Sequential(
                nn.Conv2d(3, 24, 5, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 32, 5, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 5, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(inplace=True),
                ) 
        self.speed_extractor = nn.Sequential(
                nn.Linear(1, 64),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                )
    
    def forward(self, img, speed):
        img_perception = self.img_extractor(img)
        img_perception = img_perception.view(img_perception.size(0), -1)
        spd_perception = self.speed_extractor(speed)
        x = torch.cat([img_perception, spd_perception], dim=-1)
        return x


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(1216, 64),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
                )

    def forward(self, x):
        o = self.actor(x)
        return o


class TorchDriver(nn.Module):
    def __init__(self):
        super(TorchDriver, self).__init__()
        self.perception = Perception()
        self.actor = Actor()

    def forward(self, img, speed):
        x = self.perception(img, speed)
        o = self.actor(x)
        return o
