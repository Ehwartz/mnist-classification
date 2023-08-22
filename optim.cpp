//
// Created by Ehwartz on 2022/8/13.
//

#include"optim.h"


SGD::SGD(vector<Variable*> &parameters, double lr)
{
    this->parameters = parameters;
    this->lr = lr;
}
void SGD::step()
{
    for(auto & parameter : parameters)
        *parameter -= parameter->grad * lr;
}
