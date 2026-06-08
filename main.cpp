#include "FullyConnectedLayer.hpp"
#include "Layer.hpp"
#include "MeanSquared.hpp"
#include "Model.hpp"
#include "BinaryCrossEntropy.hpp"
#include "Sigmoid.hpp"
#include "labeling_functions.hpp"
#include <cstdlib>
#include <vector>
#include <fstream>
#include <Relu.hpp>
#include <algorithm>
#include <random>
using namespace mlask;

using float_t = float;

void load_data(std::vector<std::vector<float_t>>& data){
    std::ifstream file("iris.csv");
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<float> row_features;

        for (int i = 0; i < 4; ++i) {
            std::getline(ss, token, ',');
            row_features.push_back(std::stof(token) / 5.0f - 1);
        }
        std::getline(ss, token, ',');
        if (token == "Iris-setosa"){ row_features.push_back(1);}
        else row_features.push_back(0);

        data.push_back(row_features);
    }
}

int main(){
    const int epochs = 10000;
    Model model(4,1,5, epochs, true);
    model.addLayer<FullyConnectedLayer<4,16>>();
    model.addLayer<Relu>();
    model.addLayer<FullyConnectedLayer<16,8>>();
    model.addLayer<Relu>();
    model.addLayer<FullyConnectedLayer<8,1>>();

    std::vector<std::vector<float_t>> data;

    load_data(data);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(data.begin(), data.end(), g);
    std::vector<std::vector<float_t>> train(data.begin(), data.begin()+130);
    std::vector<std::vector<float_t>> test(data.begin() + 130, data.end());

    std::vector<float_t> trainY;
    std::vector<float_t> testY;

    for (auto& row : train) {
        trainY.push_back(std::move(row.back()));
        row.pop_back();
    }
    for (auto& row : test) {
        testY.push_back(std::move(row.back()));
        row.pop_back();
    }

    model.fit<DerivedBinaryCrossEntropy>(train, trainY, epochs,  0.01f);
    ConfusionMatrix conf = model.evaluate(test, testY, post_process_sign);


    model.exportToONNX("iris_model.onnx", "iris");
    std::cout<<conf<<std::endl;
}
