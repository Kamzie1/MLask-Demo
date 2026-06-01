#include "Layer.hpp"
#include "MeanSquared.hpp"
#include "Model.hpp"
#include <cstdlib>
#include <random>
#include <iostream>
using namespace mlask;

float w = 0.3;
float b = 0;

#define SIZE 1000
#define EPOCHS 10000

int main(){
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5,0.5);
    Model model(1, 1, 1,EPOCHS, true);
    model.addFullyConnectedLayer<1, 1>();

    std::vector<float> Y;
    Y.resize(SIZE + 1);
    for(int x = 0;x<=SIZE;x++){
        Y[x] = w * x + b + distribution(generator);
    }

    float_t learning_rate = 0.001;
    for(std::size_t epochs=0; epochs < EPOCHS; epochs++){
        for(int x = 0;x<=SIZE;x++){
            model.backprop<DerivedMeanSquared>(vectorIn{{(x-(SIZE/2.f))/(SIZE/2.f)}}, vectorOut{{Y[x]}});
        }
        model.fit(learning_rate);
    }
   
    vectorIn input(SIZE+1);
    vectorOut expected(SIZE+1);
    for(int x = 0;x<=SIZE;x++){
        input(x) =  (x-(SIZE/2.f))/(SIZE/2.f);
        expected(x) =  w * x + b;
    }

    std::cout<<"Error: "<< model.whole_error<MeanSquared>(input, expected) << std::endl;
    model.exportToONNX("onnx_format.onnx", "One Neuron Neural Network");
}

