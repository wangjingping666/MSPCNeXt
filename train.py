import os
import sys
import json
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR
# import torch.optim as optim
import torch_optimizer as optim
from torchvision import transforms, datasets
import torch.nn as nn
from utils import train_and_val, plot_acc, plot_loss
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from warmup import GradualWarmupScheduler
from convnext import convnext_tiny
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
# from model_inc import inceptionnext_tiny
# from model_gai1 import inceptionnext_tiny_gai_t2
from torch_ema import ExponentialMovingAverage
from resnet50 import  resnet50
from convnext_v2 import  ConvNeXtPlusv2
# from  convnext_v1 import convnextPlusv1
# from inceptionnext import inceptionnext_tiny
# from  v2 import ConvNeXtPlusv2
from sknet import SKNet
from  v1 import convnextPlusv1

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
                                     # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder("D:/wangjingping/datasets/GrainDatasets/wheat/wheatdataset/train",
                                         transform=data_transform["train"])

    val_percent = 0.005
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, (n_train, n_val), generator=torch.Generator())  # .manual_seed(0))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)  # num_workers=4
    len_train = len(train_set)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    len_val = len(val_set)

    # net = ConvNeXtPlusv2(num_classes=7,drop_path_rate=0.6).to("cuda")
    # net = SKNet(class_num=7).to("cuda")
    net = convnextPlusv1(num_classes=7,drop_path_rate=0.6).to("cuda")
    # net = convnext_tiny(num_classes=7,drop_path_rate=0.6).to(device)
    # net = resnet50().to(device)
    # net = inceptionnext_tiny(num_classes=7,drop_path_rate=0.6,drop_rate=0.6).to("cuda")
    # net = inceptionnext_tiny_gai_t2(num_classes=7, drop_path_rate=0.6, drop_rate=0.6).to("cuda")
    #     net = inceptionnext_small(num_classes=7,drop_path_rate=0.6,drop_rate=0.6).to("cuda")
    #     net = inceptionnext_base(num_classes=7,drop_path_rate=0.6,drop_rate=0.6).to("cuda")

    #     inceptionnext_small
    #     net=convnext_tiny(num_classes=7).to("cuda")
    loss_function = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(net.parameters(), lr=0.002)
    #     optimizer = optim.Yogi(net.parameters(), lr=0.01)
    optimizer = optim.AdaMod(net.parameters(), lr=0.1)
    ema = ExponentialMovingAverage(net.parameters(), decay=0.995)

    #     multi_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    # multi_schedule = warm_up_cosine_lr_scheduler(optimizer, warm_up_epochs=10, eta_min=0.0001)
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=115, T_mult=2)
    # multi_schedule = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)
    epoch = 150
    # optimizer = optim.Adam(net.parameters(), lr=0.002)
    # mode = 'OneCycleLR'   # cosineAnn   cosineAnnWarm  OneCycleLR
    mode = 'OneCycleLR'
    max_epoch = 150
    iters = int(len_train / BATCH_SIZE if len_train / BATCH_SIZE == 0 else len_train / BATCH_SIZE + 1)
    print("max_epoch,iters", (max_epoch, iters))
    #
    if mode == 'cosineAnn':
        multi_schedule = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.001)
    elif mode == 'cosineAnnWarm':
        multi_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    elif mode == 'OneCycleLR':
        multi_schedule = OneCycleLR(optimizer, max_lr=0.003, total_steps=max_epoch * iters, pct_start=0.1,
                                    div_factor=100000, final_div_factor=0.0003)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # cur_lr_list = []
    # print(max_epoch,iters)
    # for epoch in range(max_epoch):
    #     for batch in range(iters):
    #         optimizer.step()
    #         multi_schedule.step()
    #     cur_lr = optimizer.param_groups[-1]['lr']
    #     cur_lr_list.append(cur_lr)
    #     print(cur_lr)
    # x_list = list(range(len(cur_lr_list)))
    # plt.plot(x_list, cur_lr_list)
    # plt.show()
    history = train_and_val(epoch, net, train_loader, len_train, val_loader, len_val, loss_function, optimizer,multi_schedule,
                            device,ema)
    # history = train_and_val(epoch, net, train_loader, len_train, val_loader, len_val, loss_function, optimizer,
    #                         device, ema)
    # history = train_and_val(epoch, net, train_loader, len_train,val_loader, len_val,loss_function, optimizer,device)
    plot_loss(np.arange(0, epoch), history)
    plot_acc(np.arange(0, epoch), history)