//
// Created by Ehwartz on 2022/8/9.
//

#ifndef NN_NN_H
#define NN_NN_H
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>

#include"Variable.h"
using namespace std;
using namespace cv;

class nn
{
public:
    nn* iptr;
    nn* optr;
    Variable in;
    Variable out;
    Variable grad_in;
    Variable weight;
    Variable bias;
    nn();
    virtual nn* operator()(nn* nn_i);
    virtual nn* operator()(const Variable& x);
    virtual void backward();
};

class Linear:public nn
{
public:
    Linear(int in_features, int out_features, int bias);
    virtual nn* operator()(const Variable& x);
    virtual nn* operator()(nn* nn_i);
    virtual void backward();

};

class Tanh:public nn
{
public:
    Tanh();
    virtual nn* operator()(const Variable& x);
    virtual nn* operator()(nn* nn_i);
    virtual void backward();
};

class Sigmoid:public nn
{
public:
    Sigmoid();
    virtual nn* operator()(const Variable& x);
    virtual nn* operator()(nn* nn_i);
    virtual void backward();
};

class ReLU:public nn
{
public:
    ReLU();
    virtual nn* operator()(const Variable& x);
    virtual nn* operator()(nn* nn_i);
    virtual void backward();
};

class LeakyReLU:public nn
{
public:
    LeakyReLU();
    virtual nn* operator()(const Variable& x);
    virtual nn* operator()(nn* nn_i);
    virtual void backward();
};

class CrossEntropyLoss:public nn
{
public:
    Variable Exp;
    double ExpSum;
    Variable Softmax, Log;
    CrossEntropyLoss();
    virtual nn* operator()(const Variable& input, Variable& target);
    virtual nn* operator()(nn* nn_i, Variable& target);
    virtual void backward();
};

class Softmax:public nn
{
public:
    Variable Exp;
    double ExpSum;
    Softmax();
    virtual nn* operator()(const Variable& input);
    virtual nn* operator()(nn* nn_i);
};

class MSELoss:public nn
{
    MSELoss(int size_average);
    int size_average=1;
    double sum{};
    virtual nn* operator()(const Variable& input, Variable& target);
    virtual nn* operator()(nn* nn_i, Variable& target);
    virtual void backward();
};
#endif //NN_NN_H
