# Image-Classification
mini project based on cifar-10
# Download the data package and put it in this local directory
https://pan.baidu.com/s/11XVvhuRAQE_oIb0GD4KFpQ
# Train a series of models
python main.py cfgfile#1 cfgfile#2 ...
# Visualize Predicted Samples
python classify.py name_of_model(like resnet18,AlanNet_bn) weight_file number_of_samples 
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
# training precision
![alt text](https://github.com/Ela-Boska/Image-Classification/blob/master/pictures/precision.png)
![alt text](https://github.com/Ela-Boska/Image-Classification/blob/master/pictures/precision_number.png)
# training loss
![alt text](https://github.com/Ela-Boska/Image-Classification/blob/master/pictures/training_loss.png)
![alt text](https://github.com/Ela-Boska/Image-Classification/blob/master/pictures/training_loss_number.png)
