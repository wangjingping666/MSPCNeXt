import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = torch.load("./weight/prebest_2ConvNeXtPlusv1n_0.003OneCycleLR+dp=0.6.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_correct = [0.] * 10
    class_total = [0.] * 10
    y_test, y_pred = [], []
    X_test = []

    BATCH_SIZE = 1
    data_transform = transforms.Compose([transforms.Resize([224, 224]),
                        # transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    pre_dataset = datasets.ImageFolder("D:/wangjingping/datasets/GrainDatasets/wheat/wheatdataset/test/", transform=data_transform)  # 测试集数据
    # print(len(pre_dataset))
    val_loader = torch.utils.data.DataLoader(dataset=pre_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 加载数据

    classes = pre_dataset.classes
    with torch.no_grad():
        for images, labels in val_loader:
            X_test.extend([_ for _ in images])
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            c = (predicted == labels).squeeze()
            for i, label in enumerate(labels):
                class_correct[label] += c.data.item()
                # print(class_correct)
                class_total[label] += 1
            y_pred.extend(predicted.numpy())
            y_test.extend(labels.cpu().numpy())


    print("使用的权重文件是:prebest_2ConvNeXtPlusv1n_0.003OneCycleLR+dp=0.6.pth")
    for i in range(7):
        print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:5.3f}%")


    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes)
    print("Accuracy is :", ac)
    print(cr)


    labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=labels, fmt='s', xticklabels=classes, yticklabels=classes, linewidths=0.1)
    plt.savefig('./preoutcome/acc_prebest_2ConvNeXtPlusv1n_0.003OneCycleLR+dp=0.6.png')
    plt.show()