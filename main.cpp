#include "FullyConnectedLayer.hpp"
#include "Layer.hpp"
#include "MeanSquared.hpp"
#include "Model.hpp"
#include "BinaryCrossEntropy.hpp"
#include "Sigmoid.hpp"
#include "labeling_functions.hpp"
#include <cctype>
#include <cstdlib>
#include <sstream>
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

void print_help(){
    std::cout<<"This is a demo of MLask library."<<std::endl;
    std::cout<<"The program trains the model to classify iris-setosa out of iris flowers."<<std::endl;
    std::cout<<"-f [file]: specifies to which file if any should the model be exported."<<std::endl;
    std::cout<<"-d [data]: Prints whether the data represents iris-setosa. Data should be formatted: \"n n n n\""<<std::endl;
    std::cout<<"-p whether to show progress bar(default false)."<<std::endl;
    std::cout<<"-e [number] number of epochs for training(default 10000)."<<std::endl;
    std::cout<<"-l [float] learning rate of the model(default 0.01)."<<std::endl;
}

int main(int arg, char** args){
    std::string export_file="";
    std::string data_string;
    bool run_inference = false;
    bool progress_bar = false;
    float_t learning_rate = 0.01;
    std::size_t epochs = 10000;

    if (arg<2){
        std::cout<<"Brak argumentow, zobacz -h."<<std::endl;
        exit(1);
    }

    for(int i=1;i<arg;i++){
        std::string current_arg = args[i];
        if (current_arg == "-h") {
            print_help();
            exit(0);
        }
        else if (current_arg == "-f") {
            if (i + 1 < arg) {
                export_file = args[++i];
            } else {
                std::cout << "Blad: Brak nazwy pliku po -f" << std::endl;
                exit(1);
            }
        }
        else if (current_arg == "-d") {
            if (i + 1 < arg) {
                data_string = args[++i];
                run_inference = true;
            } else {
                std::cout << "Blad: Brak danych po -d" << std::endl;
                exit(1);
            }
        }
        else if (current_arg == "-p"){
            progress_bar = true;
        }
        else if(current_arg == "-e"){
            if (i + 1 < arg) {
                epochs = std::stoi(args[++i]);
            } else {
                std::cout << "Blad: Brak danych po -e" << std::endl;
                exit(1);
            }
        }
        else if(current_arg == "-l"){
             if (i + 1 < arg) {
                 learning_rate = std::stof(args[++i]);
             } else {
                 std::cout << "Blad: Brak danych po -l" << std::endl;
                 exit(1);
             }
         }
        else {
            std::cout << "Nieznany argument: " << current_arg << std::endl;
            exit(1);
        }
    }
    Model model(4,1,5, epochs, progress_bar);
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
    std::vector<std::vector<float_t>> train(data.begin(), data.end());

    std::vector<float_t> trainY;

    for (auto& row : train) {
        trainY.push_back(std::move(row.back()));
        row.pop_back();
    }

    model.fit<DerivedBinaryCrossEntropy>(train, trainY, epochs,  learning_rate);
    if(export_file!=""){
        try{
            model.exportToONNX(args[2]);
        }
        catch(const ExportError& e){
            std::cout<<"Failed to export the model to ONNX." << std::endl;
        }
    }
    if(run_inference){
        std::vector<float_t> test;
        test.reserve(4);

        size_t size = data_string.length();
        std::string num="";
        for(size_t i=0;i<size;i++){
            if(isspace(data_string[i])){
                if(num.length()!=0){
                    test.push_back(std::stof(num));
                    num = "";
                }
            }
            else{
                num += data_string[i];
            }
        }
        if(num.length()!=0){
            test.push_back(std::stof(num));
        }
        if(test.size()!=4) {
            std::cout<<"Podales zla liczbe danych, powinny byc 4 parametry."<<std::endl;
            exit(1);
        }

        for(int i=0;i<4;i++){
            test[i] = test[i] / 5 -1;
        }

        std::stringstream input;
        input<< std::fixed << std::setprecision(3)<< model.forward(Eigen::Map<const Eigen::VectorXf>(test.data(), test.size()));
        float_t label;
        input >> label;
        if(label >= 0){
            std::cout<<"To iris-setosa!"<<std::endl;
        }
        else{
            std::cout<<"To nie jest iris-setosa!"<<std::endl;
        }
    }
}
