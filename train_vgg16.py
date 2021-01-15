import torch
import torchvision
from torchvision import datasets,transforms, models
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

path = "./datasets2"
# data transform operator  (applied to each every img at the begin)
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

# imageforder, classes are under the folder, and each class can not be empty
data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),
                                     transform = transform)
              for x in ["train", "val"]}


data_loader_image = {x:DataLoader(dataset=data_image[x],
                                                batch_size = 4,
                                                shuffle = True)
                     for x in ["train", "val"]}
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()

classes = data_image["train"].classes
classes_index = data_image["train"].class_to_idx
print(classes)
print(classes_index)
print(u"训练集个数:", len(data_image["train"]))
print(u"验证集个数:", len(data_image["val"]))

X_train, y_train = next(iter(data_loader_image["train"]))
mean = [0.5,0.5,0.5]
std  = [0.5,0.5,0.5]
img = torchvision.utils.make_grid(X_train)
img = img.numpy().transpose((1,2,0))
img = img*std+mean

print([classes[i] for i in y_train])

plt.imshow(img)
plt.show()

model = models.vgg16(pretrained=True)
#print(model)

# frozen all parameters
for parma in model.parameters():
    parma.requires_grad = False

# change the classifier layser, copy and change the last linear layer only, make the output as 2 classed
# 40-->4096
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 40),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(40,40),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(40, 2))



if use_gpu:
    model = model.cuda()


cost = torch.nn.CrossEntropyLoss()
# only optimize the model.classifier
optimizer = torch.optim.Adam(model.classifier.parameters())

#print(model)

# starts to train


n_epochs = 1

for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in tqdm(data_loader_image[param]):
            batch += 1
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            #print(y_pred, y)
            if param == "train":
                loss.backward()
                #torch.cuda.empty_cache()
                optimizer.step()
            #print(loss.data)
            #running_loss += loss.data[0] no longer fitable
            running_loss += loss.item()
            running_correct += torch.sum(pred == y.data)
            if batch % 500 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4.0 * batch), 100.0 * running_correct / (4 * batch)))

        epoch_loss = torch.true_divide(running_loss, len(data_image[param]))
        epoch_correct = 100.0 * running_correct / len(data_image[param])

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))


torch.save(model.state_dict(), "model_vgg16_finetune.pkl")
torch.save(model, "model_vgg16_finetune.pt")
data_test_img = datasets.ImageFolder(root="./datasets2/test",
                                     transform = transform)
data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img,
                                                   batch_size = 16)

image, label = next(iter(data_loader_test_img))
images = Variable(image.cuda())

y_pred = model(images)
_, pred = torch.max(y_pred.data, 1)
print(pred)

img = torchvision.utils.make_grid(image)
img = img.numpy().transpose(1,2,0)
mean = [0.5,0.5,0.5]
std  = [0.5,0.5,0.5]
img = img*std+mean
print("Pred Label:",[classes[i] for i in pred])
plt.imshow(img)
plt.show()