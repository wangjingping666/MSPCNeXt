import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from predict import pre
import os
# ./weight/last1_5Block_d_1.CosineAnnealingLR=0.0001+dp=0.5.pth
str_path_model = "./weight/"
str_path_model_1 = "_3ConvNeXtPlusv1Q"
str_path_model_2 = "_0.003OneCycleLR+dp=0.6.pth"


def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,multi_schedule,device,ema):
# def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,device,ema):

    torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc_t = 0
    best_acc_v = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                # forward
                output = model(image)
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]
                # print(predict_t)
                # backward
                loss.backward()
                optimizer.step()  # update weight
                ema.update()
                if e<150:
                    multi_schedule.step()
                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()
                # print(training_acc)
                pbar.update(1)

#             multi_schedule.step()


        model.eval()   # 自动修改train为False
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]


                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculatio mean for each batch
            train_loss.append(running_loss / len_train)
            val_loss.append(val_losses / len_val)

            train_acc.append(training_acc / len_train)
            val_acc.append(validation_acc / len_val)

            torch.save(model, str_path_model + "last" + str_path_model_1 + str_path_model_2)
            if best_acc_v < (validation_acc / len_val):
                best_acc_v = validation_acc / len_val

                torch.save(model, str_path_model + "best" + str_path_model_1 + str_path_model_2)
            print("第%d个epoch的学习率：%f" % (e + 1, optimizer.param_groups[0]['lr']))

            ac_test = pre(str_path_model + "last" + str_path_model_1 + str_path_model_2)
            if best_acc_t < ac_test:  # 保存在测试集上最好的结果
                best_acc_t = ac_test

                torch.save(model, str_path_model + "prebest" + str_path_model_1 + str_path_model_2)



            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Acc: {:.5f}..".format(training_acc / len_train),
                  "Val Acc: {:.5f}..".format(validation_acc / len_val),
                  "Train Loss: {:.3f}..".format(running_loss / len_train),
                  "Val Loss: {:.3f}..".format(val_losses / len_val),
                  "Time: {:.2f}s".format((time.time() - since)),
                  "Test Acc: {:.5f}..".format(best_acc_t))



    history = {'train_loss': train_loss, 'val_loss': val_loss ,'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history

def plot_loss(x, history):
    plt.plot(x, history['val_loss'], label='val',color='b', marker='o')
    plt.plot(x, history['train_loss'], label='train',color='r', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()

    plt.savefig((str_path_model + "loss" + str_path_model_1 + str_path_model_2).replace(".pth",".png"))
    plt.show()


def plot_acc(x, history):
    plt.plot(x, history['train_acc'], label='train_acc',color='r', marker='x')
    plt.plot(x, history['val_acc'], label='val_acc', color='b',marker='x')
    plt.title('Acc per epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig((str_path_model + "acc" + str_path_model_1 + str_path_model_2).replace(".pth",".png"))
    plt.show()