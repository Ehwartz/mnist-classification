//
// Created by Ehwartz on 2022/8/9.
//
#include"nn.h"

//// nn
nn::nn()
{
    iptr = nullptr;
    optr = nullptr;
}
nn* nn::operator()(const Variable& x)
{
    return this;
}
nn* nn::operator()(nn* nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}
void nn::backward()
{
    cout<<"nn::backward"<<endl;
    if(this->iptr)
    {
        this->iptr->backward();
    }
}

//// Linear
Linear::Linear(int in_features, int out_features, int bias):
nn()
{
    this->weight=Variable(in_features, out_features);
    this->bias=Variable(1, out_features);
    weight.grad = Mat::zeros(in_features, out_features, CV_64FC1);
    grad_in = Variable::ones(1, in_features, CV_64FC1);
    if(!bias)
    {
        this->bias *= 0;
        this->bias.grad *= 0;
    }
}
nn* Linear::operator()(const Variable& x)
{
    in = x;
    out = (x * this->weight + this->bias);
    return this;
}
 nn* Linear::operator()(nn* nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}
void Linear::backward()
{
    if(this->optr)
    {
        this->weight.grad += this->in.t() * this->optr->grad_in;
        this->bias.grad += this->optr->grad_in;
        this->grad_in = this->optr->grad_in * this->weight.t();
    }
    if(this->iptr)
        this->iptr->backward();
}

//// Tanh
Tanh::Tanh() = default;
nn* Tanh::operator()(const Variable &x)
{
    in = x;
    exp(x, out);
    out = (out-1/out)/(out+1/out);
    this->grad_in = 1-out.mul(out);
    return this;
}
nn* Tanh::operator()(nn* nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}
void Tanh::backward()
{
    if(this->optr)
        this->grad_in = this->grad_in.mul(this->optr->grad_in);
    if(this->iptr)
        this->iptr->backward();
}

//// Sigmoid
Sigmoid::Sigmoid() = default;
nn* Sigmoid::operator()(const Variable &x)
{
    in = x;
    exp(x, out);
    out = 1/(1+1/out);
    this->grad_in = out.mul(1-out);
    return this;
}
nn* Sigmoid::operator()(nn *nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}
void Sigmoid::backward()
{
    if(this->optr)
        this->grad_in = this->grad_in.mul(this->optr->grad_in);
    if(this->iptr)
        this->iptr->backward();
}

//// ReLU
ReLU::ReLU() = default;
nn* ReLU::operator()(const Variable &x)
{
    in = x;
    out = 0.5*x + 0.5*abs(x);
    divide(out, x, grad_in);
}
nn* ReLU::operator()(nn *nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}
void ReLU::backward()
{
    if(this->optr)
        this->grad_in = this->grad_in.mul(this->optr->grad_in);
    if(this->iptr)
        this->iptr->backward();
}

//// LeakyReLU
LeakyReLU::LeakyReLU() = default;
nn* LeakyReLU::operator()(const Variable &x)
{
    in = x;
    out = 0.55*x + 0.45*abs(x);
    divide(out, x, grad_in);
}
nn* LeakyReLU::operator()(nn *nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}
void LeakyReLU::backward()
{
    if(this->optr)
        this->grad_in = this->grad_in.mul(this->optr->grad_in);
    if(this->iptr)
        this->iptr->backward();
}

//// CrossEntropyLoss
CrossEntropyLoss::CrossEntropyLoss():
Exp(1, 1),
ExpSum(0),
Softmax(Exp),
Log(Exp){}
nn* CrossEntropyLoss::operator()(const Variable& input, Variable& target)
{
    this->in = input;
    exp(input, Exp);
    ExpSum = sum(Exp)[0];
    Softmax = Exp/ExpSum;
    log(Softmax, Log);
    out = - Log * target.t();
    this->grad_in = Softmax - target;
    return this;
}
nn* CrossEntropyLoss::operator()(nn *nn_i, Variable& target)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out, target);
    return this;
}
void CrossEntropyLoss::backward()
{
    if(this->optr)
        this->grad_in = this->grad_in.mul(this->optr->grad_in);
    if(this->iptr)
        this->iptr->backward();
}

//// Softmax
Softmax::Softmax():
Exp(1, 1),
ExpSum(0){}
nn* Softmax::operator()(const Variable &input)
{
    this->in = input;
    exp(input, Exp);
    ExpSum = sum(Exp)[0];
    this->out=Exp/ExpSum;
    return this;
}
nn* Softmax::operator()(nn *nn_i)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out);
    return this;
}

//// MSELoss
MSELoss::MSELoss(int size_average)
{
    this->size_average = size_average;
}
nn* MSELoss::operator()(const Variable &input, Variable &target)
{
    in = input;
    out = (input-target) * (input-target).t();
    if(size_average)
    {
        int size = input.cols;
        out /= out;
    }
    grad_in = 2 * (input -target);
    return this;
}
nn* MSELoss::operator()(nn *nn_i, Variable &target)
{
    this->iptr=nn_i;
    nn_i->optr=this;
    this->operator()(nn_i->out, target);
    return this;
}
void MSELoss::backward()
{
    if(this->optr)
        this->grad_in = this->grad_in.mul(this->optr->grad_in);
    if(this->iptr)
        this->iptr->backward();
}