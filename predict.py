import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.autograd import Variable

transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

'''
model = models.vgg16(pretrained=True)
#print(model)

# frozen all parameters
for parma in model.parameters():
    parma.requires_grad = False

# change the classifier layser, copy and change the last linear layer only, make the output as 2 classed
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 40),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(40,40),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(40, 2))


#model.load_state_dict("model_vgg16_finetune.pkl")
#model.load_state_dict(torch.load('model_vgg16_finetune.pkl', map_location='gpu'))
print(model)
'''

model = torch.load('./checkpoints/model_vgg16_finetune.pt')
data_test_img = datasets.ImageFolder(root="./datasets2/test2",
                                     transform = transform)
data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img,
                                                   batch_size = 16)

print()
classes = ['cats', 'dogs']
classes_index = [0, 1]
# get a bach sample
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