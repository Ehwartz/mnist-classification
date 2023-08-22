//
// Created by Ehwartz on 2022/8/13.
//

#ifndef NN_OPTIM_H
#define NN_OPTIM_H
#include"Variable.h"


class SGD
{
public:
    vector<Variable*>parameters;
    double lr;
    SGD(vector<Variable*>&parameters, double lr);
    void step();
};


#endif //NN_OPTIM_H
