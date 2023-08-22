//
// Created by Ehwartz on 2022/8/13.
//

#ifndef NN_MODULE_H
#define NN_MODULE_H
#include<vector>
#include"nn.h"
#include<sstream>
#include<cstring>
class Module
{
public:
    Module();
    vector<Variable*>parameters;
    Variable out;
    void init();
    nn* ptr;
    Linear linear1=Linear(784, 512, 1);
    Tanh tanh1;
    Linear linear2=Linear(512, 256, 1);
    Tanh tanh2;
    Linear linear3=Linear(256, 10, 1);
    nn* forward(Variable x);
    nn* operator()(Variable x);
    nn* operator()(nn* nn_i);
    void zero_grad();
    void save_weights(string &path);
    void load_weights(string &path);

};

#endif //NN_MODULE_H
