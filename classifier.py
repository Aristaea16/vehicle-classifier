##############################################################
## Author: James Ortiz
## File: classifier.py
## DataSet: CIFAR10 (10,000 images, including cars)
## Version: 1.0
## Description: Convolutional-Neural-Net image classifier that
## can predict if the image is 1 of 10 different classes
## (plane, bird, car, deer, dog, cat, from, horse, ship,
## truck)
## --Modified to identify car images
## --of the 10 different image classes.
## Prerequisites:
## 1) numpy installation (can be done via Anaconda)
## 2) PyTorch installation, can be installed through conda
##############################################################


import torch
import torchvision
import torchvision.transforms as transforms

#Normalize Tensor

transform  = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))])

#Download training set:
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
#Download testing set:
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

#Classes used to identify the pictures in the dataset:

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

#function to show an image

def imshow(img):
    img  = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images

dataiter = iter(trainloader)
images, labels = dataiter.next()

#show images
#imshow(torchvision.utils.make_grid(images))

#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#------------------------------------------------------

#Build Neural-Net Skeleton:

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 27, 5) #Increase width of network 2nd arg:
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(27, 16, 5) #Increase width of network 1st arg:
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Define a Loss Function and Optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#Define the network (currently 24 epochs)
#Changed from range(2) -> 24
for epoch in range(14):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs:
        # data is a list of [inputs, labels]

        inputs, labels = data

        #zero the parameter gradients

        optimizer.zero_grad()

        #forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Test the network on the test data
# Network has been successfully trained for 2 full passes
# over the dataset.

# First, display an image from the test:

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images:

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# How the network performs on the whole dataset:

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

#The average accuracy of the identification of all 10 image classes:

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Find out which classes performed well, and which didn't:
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

#EDITED: used to be range(10):
#for i in range(10)

#for i in range(10)

#Printing only the accuracy of cars, out of the total 10 types of images:
print('Accuracy of %5s : %2d %%' % (
        classes[1], 100 * class_correct[1] / class_total[1]))
