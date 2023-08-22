//
// Created by Ehwartz on 2022/8/14.
//

#ifndef NN_DATASET_H
#define NN_DATASET_H
#include<opencv2/opencv.hpp>
#include<cstring>
#include<iostream>
#include<fstream>
#include<ctime>
using namespace std;
using namespace cv;

class dataset
{
public:
    dataset(string& root);

    vector<Mat>data;
    vector<Mat>labels;
};

#endif //NN_DATASET_H
