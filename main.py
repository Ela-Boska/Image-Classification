# -*- coding: utf-8 -*-
import torch.nn as nn
import time,math,torch,AlexNet,preprocess,sys
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorboardX import SummaryWriter
from writer import *

cfgfiles = sys.argv[1:]
for cfgfile in cfgfiles:
    arguments = preprocess.parse(cfgfile)

    n_classes = 10
    change_point = arguments['milestones']
    lrs = arguments['lrs']
    num_workers = arguments['num_workers']
    batch_size = arguments['batch_size']
    print('num workers =',num_workers)
    print('batch size =',batch_size)
    loss_func = nn.CrossEntropyLoss()
    log = arguments['log_directory']
    writer = SummaryWriter(log)
    MODEL = arguments['model']
    model = getattr(AlexNet,MODEL)()
    log_graph(writer,model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=lrs[0],betas=(0.9,0.999),weight_decay=0.0)


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_file_list = ['cifar10/data_batch_1','cifar10/data_batch_2','cifar10/data_batch_3','cifar10/data_batch_4','cifar10/data_batch_5']
    val_file_list = ['cifar10/test_batch']
    length = len(preprocess.dataset(train_file_list))
    dataloader = torch.utils.data.DataLoader(preprocess.dataset(train_file_list,transform=transform_train),
        shuffle =True,batch_size=batch_size,num_workers=num_workers)

    val_dataloader = torch.utils.data.DataLoader(preprocess.dataset(val_file_list,transform=transform_test),
        shuffle =True,batch_size=batch_size,num_workers=num_workers)


    model.train()
    best_pre = 0
    if arguments['pretrained_weight']:
        model.load_weights(arguments['pretrained_weight'])
        best_pre = model.evaluate(val_dataloader)
        i = 0
        while  i<len(change_point) and change_point[i]<=model.epoch:
            i+=1
        optimizer = torch.optim.Adam(model.parameters(),lr=lrs[i],betas=(0.9,0.999))

    last_time = time.time()
    max_epoch=arguments['max_epoch']


    last = time.time()
    last_data = time.asctime( time.localtime(last) )
    print(last_data,'epoch',model.epoch,'lr =',optimizer.state_dict()['param_groups'][0]['lr'])
    print()

    count=0
    count_per_epoch = math.ceil(len(dataloader)//batch_size)
    init_epoch = model.epoch
    while model.epoch < max_epoch:
        
        if model.epoch in change_point:
            i = 0
            while i<len(change_point) and change_point[i]<=model.epoch:
                i+=1
            optimizer = torch.optim.Adam(model.parameters(),lr=lrs[i],betas=(0.9,0.999))
        for batch,labels in dataloader:
            labels = labels.cuda()
            out = model(batch.cuda())
            loss = loss_func(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_id = count+init_epoch*count_per_epoch
            writer.add_scalar('training loss',loss.item(),batch_id)
            count+=1
        model.epoch +=1
        time1 = time.time()
        precision,val_loss = model.evaluate(val_dataloader)
        time2 = time.time()
        writer.add_scalar('precision',precision,batch_id)
        writer.add_scalar('val_loss',val_loss,batch_id)
        now = time.time()
        now_date = time.asctime( time.localtime(now) )
        span = now-last; last = now
        speed = length/span
        if arguments['log_parameters']:
            log_weights(writer,model,batch_id)
        print(now_date,'speed =',speed,'images/sec','time used =',span/60,'min','lr =',optimizer.state_dict()['param_groups'][0]['lr'])
        print()
        if precision>best_pre:
            best_pre = precision
            model.save_weights(arguments['weights_savefile'])
    


