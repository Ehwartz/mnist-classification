//
// Created by Ehwartz on 2022/8/22.
//

#ifndef NN_STAT_H
#define NN_STAT_H
#include<iostream>
#include<cstring>
#include<ctime>
#include"Variable.h"
#include"Module.h"
#include"optim.h"
#include"nn.h"
#include"dataset.h"
double StatCorrect(Module& model, dataset& Dataset);


#endif //NN_STAT_H
