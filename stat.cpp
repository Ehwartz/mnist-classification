//
// Created by Ehwartz on 2022/8/22.
//
#include"stat.h"
double StatCorrect(Module& model, dataset& Dataset)
{
    int total = Dataset.data.size();
    int correct = 0;
    int maxID[2], maxID_t[2];
    Softmax softmax;
    for(int i=0;i<total;++i)
    {
        softmax(model(Dataset.data[i]));
        minMaxIdx(softmax.out, nullptr, nullptr, nullptr, maxID);
        minMaxIdx(Dataset.labels[i], nullptr, nullptr, nullptr, maxID_t);
        if(maxID[0]==maxID_t[0] && maxID[1]==maxID_t[1])
        {
            ++correct;
            cout<<"ID: "<<i<<": Correct"<<endl;
        }
        else
            cout<<"ID: "<<i<<": Wrong"<<endl;

    }
    double accuracy = (double)correct/total;
    return accuracy;
}