# -*- coding: utf-8 -*-
import preprocess
import AlexNet
import sys



def detect(model='resnet18'):
    model = getattr(AlexNet,model)()
    model.load_weights()
    dataset = preprocess.dataset(['cifar10/test_batch'])
    AlexNet.show_sample(model,dataset)

if __name__ == "__main__":
    detect(sys.argv[1])
