#include <iostream>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<ctime>


#include"nn.h"
#include"Variable.h"
#include"Module.h"
#include"optim.h"
#include"dataset.h"
#include"train.h"
#include"stat.h"
using namespace std;
using namespace cv;

int main()
{

    double start = clock();
    Module model;
    // model.init();
    CrossEntropyLoss criterion;
    SGD optimizer(model.parameters, 0.0001);
    string train_root = "../mnist/train";
    string test_root = "../mnist/test";
    string path = "../weights";
    model.load_weights(path);
    dataset Dataset(test_root);
//     Softmax softmax;
    int epoch=25;
    int batch_size=100;
    train(model, Dataset, criterion, optimizer, epoch, batch_size, train_root, path);
    double accuracy = StatCorrect(model, Dataset);
    cout<<"Accuracy: "<<accuracy*100<<"%"<<endl;


    return 0;
}
