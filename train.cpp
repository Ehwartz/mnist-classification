//
// Created by Ehwartz on 2022/8/17.
//
#include "train.h"


void train(Module& model, dataset& Dataset, CrossEntropyLoss& criterion, SGD optimizer, int epoch, int batch_size, string root, string path)
{
    double start = clock();
    nn* ptr;
    int batch_num = Dataset.data.size()/batch_size;
    for(int ep=0;ep<epoch;++ep)
    {
        double loss_sum = 0;
        model.zero_grad();
        double t = clock();
        for (int index = 0; index < Dataset.data.size(); ++index)
        {

            ptr = model(Dataset.data[index]);
            ptr = criterion(ptr, (Variable &) Dataset.labels[index]);
            ptr->backward();
            loss_sum += criterion.out.at<double>(0, 0);
            if ((index + 1) % batch_size == 0)
            {
                optimizer.step();
                cout << "ep: " << ep <<"/"<<epoch<<"\t"
                     << "Batch Index:" << int(index / batch_size)<<"/" <<batch_num<<"  \t"
                     << "Loss: " << loss_sum<<"\t";
                loss_sum = 0;
                model.zero_grad();
                cout<<"BatchTime: "<<(clock()-t)/1000<<endl;
                t = clock();
            }
        }
    }
    model.save_weights(path);
    double end = clock();
    cout<<"Training time: "<<((int)(end-start))/1000<<"  seconds"<<endl;
}
