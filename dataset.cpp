//
// Created by Ehwartz on 2022/8/14.
//
#include"dataset.h"

dataset::dataset(string& root)
{
    fstream f;
    f.open(root+".txt");
    // cout<<f.is_open()<<endl;
    string line;
    string img_name;
    int label;
    cout<<"Loading images..."<<endl;
    double t1 = clock();
    while(getline(f, line))
    {
        istringstream elements(line);
        elements>>img_name;
        elements>>label;
        Mat target=Mat::zeros(1, 10, CV_64FC1);
        target.at<double>(0, label) = 1;
        Mat img = imread(root+"/"+img_name);
        cvtColor(img, img, COLOR_BGR2GRAY);
        img.convertTo(img, CV_64FC1);
        normalize(img, img, 1.0, 0, NORM_MINMAX);
        data.push_back(img.reshape(1, 1));
        labels.push_back(target);
        cout<<"Loading image: "<<data.size()<<endl;
        if(data.size()>2500) break;
    }
    double t2 = clock();
    printf("Time of Loading: %.3f  seconds\n",(t2-t1)/1000);

}