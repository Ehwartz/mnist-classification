//
// Created by Ehwartz on 2022/8/13.
//

#include"Module.h"
Module::Module() {init();}
void Module::init()
{
    Variable x(1, linear1.weight.rows);
    forward(x);
    while(ptr)
    {
        if(!ptr->weight.empty())
        {
            parameters.push_back(&ptr->weight);
            parameters.push_back(&ptr->bias);
        }
        ptr=ptr->iptr;
    }
}

nn* Module::forward(Variable x)
{
    ptr = linear1(x);
    ptr = tanh1(ptr);
    ptr = linear2(ptr);
    ptr = tanh2(ptr);
    ptr = linear3(ptr);
    this->out = ptr->out;
    return ptr;
}

nn* Module::operator()(Variable x)
{
    return forward(x);
}
nn* Module::operator()(nn* nn_i)
{
    return forward(nn_i->out);
}
void Module::zero_grad()
{
    for(auto & parameter : parameters)
        parameter->grad *= 0;
}
void Module::save_weights(string &path)
{
    int index = 0;
    for(auto & parameter: parameters)
    {
        ostringstream file_path;
        file_path<<path<<"/parameter"<<index<<".xml";
        FileStorage fs(file_path.str(), FileStorage::WRITE);
        fs<<"weight"<<*parameter;
        fs.release();
        ++index;
    }
}
void Module::load_weights(string &path)
{
    int index = 0;
    for(auto & parameter: parameters)
    {
        ostringstream file_path;
        file_path<<path<<"/parameter"<<index<<".xml";
        FileStorage fs(file_path.str(), FileStorage::READ);
        fs["weight"]>>*parameter;
        fs.release();
        ++index;
    }
}