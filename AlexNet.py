# -*- coding: utf-8 -*-
import torch
from torchvision import datasets,transforms,models
from torch import nn
import random
import matplotlib.pyplot as plt
import math

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def show_sample(model,dataset,use_cuda=False):
    image,label = dataset[random.randint(0,len(dataset)-1)]
    img = transform_test(image)
    img = img.view(1,*img.shape)
    if use_cuda:
        img = img.cuda()
    out = model(img)
    class_id = out.argmax(-1).item()
    name = dataset.names[class_id]
    label = dataset.names[label]
    print(name,label)
    plt.imshow(image)
    plt.show()

def reorgan(inputs,scale=2):
    N,C,H,W = inputs.shape
    inputs = inputs.view(N,C,H//scale,scale,W//scale,scale).transpose(-2,-3)
    inputs = inputs.contiguous().view(N,C,H*W//(scale**2),scale**2)
    inputs = inputs.transpose(-1,-2).contiguous().view(N,C*scale**2,H//scale,W//scale)
    return inputs


def evaluate(model,val_dataloader):
    loss_func = nn.CrossEntropyLoss()
    precision = 0
    count = 0
    loss_total = 0
    model.eval()
    for batch,labels in val_dataloader:
        out = model(batch.cuda())
        labels = labels.cuda()
        ans = out.argmax(-1)
        precision += torch.mean((ans==labels).float()).item()
        loss_total += loss_func(out,labels).item()
        count += 1
    model.train()
    precision /= count
    loss_total /= count
    print('epoch {0}, precision = {1}%'.format(model.epoch, 100*precision))
    return precision,loss_total

class MYNet(nn.Module):
    def save_weights(self,file):
        torch.save((self.state_dict(),self.epoch),file)

    def load_weights(self,file):
        data =torch.load(file)
        self.load_state_dict(data[0])
        self.epoch = data[1]

    def evaluate(self,dataset):
        return evaluate(self,dataset)

    def initialize(self):
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class AlexNet(MYNet):

    def __init__(self,input_size=(3,32,32), num_classes=10):
        super(AlexNet, self).__init__()
        self.epoch = 0
        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=11, stride=4, padding=5),      #input/4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding = 1),          #input/8
            nn.Conv2d(64, 192, kernel_size=5, padding=2),               
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding = 1),          #input/16
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256 ,kernel_size=2, stride=2),               #input/32
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 , 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.initialize()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x


class NaiveNet(MYNet):

    def __init__(self,input_size=(3,32,32), num_classes=10):
        super(NaiveNet, self).__init__()
        self.epoch = 0
        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1),      #input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding = 0),          #input/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),                 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding = 0),          #input/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512 ,kernel_size=2, stride=2),               #input/8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*input_size[1]*input_size[2]//64 , 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.initialize()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class AlanNet(MYNet):

    def __init__(self,input_size=(3,32,32), num_classes=10):
        super(AlanNet, self).__init__()
        self.epoch = 0
        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1),      #input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding = 0),          #input/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),                 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding = 0),          #input/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512 ,kernel_size=2, stride=2),               #input/8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*input_size[1]*input_size[2]//64 , 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.initialize()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet(MYNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.epoch = 0
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,   
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # input/2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # input/4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)     # input/8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)     # input/16
        self.fc = nn.Linear(512 * block.expansion*4, num_classes)

        self.initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_original(MYNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.epoch = 0
        super(ResNet_original, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class RRResNet(MYNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.epoch = 0
        super(RRResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,   
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU (init=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # input/2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # input/4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)     # input/8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)     # input/16
        self.fc = nn.Linear(512 * block.expansion*4, num_classes)

        self.initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class RResNet(MYNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.epoch = 0
        super(RResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,   
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # input/2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # input/4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)     # input/8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)     # input/16
        self.fc = nn.Linear(512 * block.expansion*4*3, num_classes)

        self.initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        cut = self.layer3(x)
        x = self.layer4(cut)
        cut = reorgan(cut)
        x = torch.cat([x,cut],1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(init=0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model

def rresnet18(pretrained=False, **kwargs):
    """Constructs a RResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model

def rrresnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RRResNet(PBasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model

def resnet18_original(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_original(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model