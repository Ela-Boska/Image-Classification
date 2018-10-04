# Image-Classification
mini project based on cifar-10
# Main Usage
python main.py #cfgfile#
# environment
python3 pytorch 0.4.0
# preprocess
random crop
random flip
normalize based on std and mean
# features
use tensorboardX to record loss and precision during training.
# networks
original AlexNet and resnet18 copied from source code of pytorch  
altered version of AlexNet: NaiveNet and AlanNet  
altered version of resnet18: rresnet18 and rrresnet18
