//
// Created by Ehwartz on 2022/8/9.
//
#include "Variable.h"

#include <utility>
Variable::Variable():Mat()
{

}

Variable::Variable(int in_features, int out_features):Mat(in_features, out_features, CV_64FC1)
{
    RNG rng;
    rng.fill(*this, RNG::UNIFORM, -1.f, 1.f);
    this->grad = Mat::ones(in_features, out_features, CV_64FC1);
}
Variable::Variable(Mat m):Mat(std::move(m))
{}
Variable::Variable(MatExpr m): Variable((Mat)m)
{}

Variable::Variable(Variable& v): Variable(Mat(v))
{
  this->grad = v.grad;
}

Variable& Variable::operator=(Variable const& v)
{
    Mat::operator=(v);
    this->grad=v.grad;
}
Variable& Variable::operator=(const MatExpr& m)
{
    Mat::operator=(m);
}

