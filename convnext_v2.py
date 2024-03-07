import os
import sys
import json
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR
import torch_optimizer as optim
from torchvision import transforms, datasets
import torch.nn as nn
from utils import train_and_val, plot_acc, plot_loss
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from torch_ema import ExponentialMovingAverage
from resnet50 import resnet50
from MSPCNeXt_v1 import mspcnext_v1
from MSPCNeXt_v2 import mspcnext_v2

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

if __name__ == '__main__':
    # on_delete = nn.Module.CASCADE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 64

    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder("Training set path",
                                         transform=data_transform["train"])

    val_percent = 0.2
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, (n_train, n_val), generator=torch.Generator())
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    len_train = len(train_set)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    len_val = len(val_set)

    net = mspcnext_v1(num_classes=7,drop_path_rate=0.6).to("cuda")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdaMod(net.parameters(), lr=0.1)
    ema = ExponentialMovingAverage(net.parameters(), decay=0.995)

    epoch = 150
    mode = 'OneCycleLR'
    max_epoch = 150
    iters = int(len_train / BATCH_SIZE if len_train / BATCH_SIZE == 0 else len_train / BATCH_SIZE + 1)
    print("max_epoch,iters", (max_epoch, iters))

    if mode == 'cosineAnn':
        multi_schedule = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.001)
    elif mode == 'cosineAnnWarm':
        multi_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    elif mode == 'OneCycleLR':
        multi_schedule = OneCycleLR(optimizer, max_lr=0.003, total_steps=max_epoch * iters, pct_start=0.1,
                                    div_factor=100000, final_div_factor=0.0003)

    history = train_and_val(epoch, net, train_loader, len_train, val_loader, len_val, loss_function, optimizer,multi_schedule,device,ema)

    plot_loss(np.arange(0, epoch), history)
    plot_acc(np.arange(0, epoch), history)
