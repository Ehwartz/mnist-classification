//
// Created by Ehwartz on 2022/8/9.
//

#ifndef NN_VARIABLE_H
#define NN_VARIABLE_H
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
using namespace std;
using namespace cv;
class Variable:public Mat
{
public:
    Mat grad;

    //__attribute__((unused)) Variable(Mat m);
    Variable();
    Variable(int in_features, int out_features);
    Variable(Mat m);
    Variable(MatExpr m);
    Variable(Variable& v);
    //Variable& operator=(const Mat& m);
    Variable& operator=(Variable const& v);
    Variable& operator=(const MatExpr& m);

};


#endif //NN_VARIABLE_H
