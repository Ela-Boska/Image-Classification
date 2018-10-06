# -*- coding: utf-8 -*-
import preprocess
import AlexNet
import sys



def detect(model='resnet18',weight=None,num=1):
    model = getattr(AlexNet,model)()
    model.load_weights(weight)
    dataset = preprocess.dataset(['cifar10/test_batch'])
    AlexNet.show_sample(model,dataset,num)

if __name__ == "__main__":
    detect(sys.argv[1],sys.argv[2],int(sys.argv[3]))
