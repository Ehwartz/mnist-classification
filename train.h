//
// Created by Ehwartz on 2022/8/17.
//

#ifndef NN_TRAIN_H
#define NN_TRAIN_H
#include<iostream>
#include<cstring>
#include<ctime>
#include"Variable.h"
#include"Module.h"
#include"optim.h"
#include"nn.h"
#include"dataset.h"

void train(Module& model,
           dataset& Dataset,
           CrossEntropyLoss& criterion,
           SGD optimizer,
           int epoch,
           int batch_size,
           string root,
           string path);



#endif //NN_TRAIN_H
