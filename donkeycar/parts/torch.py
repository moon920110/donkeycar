'''

pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion. 

'''


import os
import time
import copy
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import donkeycar as dk
from torch.optim import Adam, SGD
from tqdm import trange, tqdm
from PIL import Image


def numpy_to_pil(img_):
    img = copy.deepcopy(img_)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img *= 255
    img = img.astype('uint8')

    im_obj = Image.fromarray(img)
    return im_obj


class TorchPilot(object):
    '''
    Base class for Torch models that will provide steering and throttle to guide a car.
    '''
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.best_state_dict = None
 
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save(self, model_path):
        torch.save(self.best_state_dict, model_path)
        
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
        self.best_state_dict = copy.deepcopy(self.model.state_dict())

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
                    if val_losses < 0.1:
                        if val_losses < min_val_loss:
                            no_improve_cnt = 0
                            min_val_loss = val_losses
                            self.best_state_dict = copy.deepcopy(self.model.state_dict())
                        else:
                            no_improve_cnt += 1

                t.set_postfix(avg_loss=losses/train_step)

                if no_improve_cnt == 6:
                    print("EARLY STOP: {}".format(min_val_loss))
                    self.save(model_name)
                    break
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        self.save(model_name)

    
class TorchIL(TorchPilot):
    '''
    Pretrain network before RL using IL
    '''
    def __init__(self, height=120, width=160, *args, **kwargs):
        super(TorchIL, self).__init__(*args, **kwargs)
        self.model = TorchDriver()
        self.transforms = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            ])

    def run(self, img_arr, speed):
        self.model.eval()
        img = numpy_to_pil(img_arr)
        img = self.transforms(img).unsqueeze(0)
        outputs = self.model(img,torch.Tensor([[speed]]))

        return outputs[0][0], outputs[0][1]


class Perception(nn.Module):
    def __init__(self):
        super(Perception, self).__init__()
        drop = 0.5
        self.img_extractor = nn.Sequential(
                nn.Conv2d(3, 24, 5, 2),
                nn.ReLU(),
                #nn.Dropout2d(drop),
                nn.Conv2d(24, 32, 5, 2),
                nn.ReLU(),
                #nn.Dropout2d(drop),
                nn.Conv2d(32, 64, 5, 2),
                nn.ReLU(),
                #nn.Dropout2d(drop),
                nn.Conv2d(64, 64, 3, 2),
                nn.ReLU(),
                #nn.Dropout2d(drop),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                #nn.Dropout2d(drop),
                ) 
        self.speed_extractor = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(drop),
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
        drop = 0.5
        self.actor = nn.Sequential(
                nn.Linear(1216, 100),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(50, 2),
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
