#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "../matrix.hpp"
#include "../nn.hpp"
#include <algorithm>
#include <ctime>
#include <cstdlib>

using namespace std;

vector<string> split(const string& s, char delimiter) {
  vector<string> tokens;
  string token;
  istringstream tokenStream(s);
  while (getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

Matrix getTarget(const std::string& s){
  int _class;

  if(s == "Iris-setosa"){
    _class = 0;
  } else if(s == "Iris-versicolor") {
    _class = 1;
  } else if(s == "Iris-virginica") {
    _class = 2;
  } else {
    throw "unidentified class";
  }
  Matrix result = Matrix(3,1,"target");
  result.setZeros();
  result.at(_class, 0) = 1;
  return result;
}

vector<float> getTargetVector(const std::string& s){
  int _class;
  vector<float> result(3,0);
  if(s == "Iris-setosa"){
    _class = 0;
  } else if(s == "Iris-versicolor") {
    _class = 1;
  } else if(s == "Iris-virginica") {
    _class = 2;
  } else {
    throw "unidentified class";
  }
  result[_class] = 1;
  return result;
}

int main() {
  // srand(42);  
  std::srand ( unsigned ( std::time(0) ) );
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");
  USE_MATRIX_NAMES = false;

  try {
    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(4);
    fnn.addLayer(4);
    fnn.addLayer(10);
    fnn.addLayer(3);
    fnn.compile();
    float lr = 0.00005;

    string line;
    spdlog::info("Training start");

    // SINGLE SAMPLE SGD
    // for (int e = 0; e < 20; ++e) {
    //   spdlog::info("Epoch {}", e);
    //   ifstream trainData;
    //   trainData.open("../data/iris_train.txt");
    //   int trainingStepCounter = 0;
    //   while(getline(trainData, line)){
    //     vector<string> tokens = split(line, ',');
    //     Matrix target = getTarget(tokens[tokens.size() - 1]);
    //     tokens.pop_back();
    //     Matrix input(tokens.size(),1,"input");
    //     input.setZeros();
    //     for(int i = 0; i < tokens.size(); i++) {
    //       input.at(i, 0) = stof(tokens[i]);
    //     }
    //     fnn.stepTrain(input,target,lr);
    //     /* spdlog::info("Training step {} complete", trainingStepCounter); */
    //     trainingStepCounter++;
    //   }
    //   trainData.close();
    // }

    // MINI BATCH SGD
    ifstream trainDf;
    trainDf.open("../data/iris_train.txt");
    vector< vector<float> > input,target;
    while(getline(trainDf, line)){
      vector<string> tokens = split(line, ',');
      vector<float> tgt = getTargetVector(tokens[tokens.size() - 1]);
      tokens.pop_back();
      vector<float> inp(tokens.size(),0);
      for(int i = 0; i < tokens.size(); i++) {
        inp[i] = stof(tokens[i]);
      }
      input.push_back(inp);
      target.push_back(tgt);
    }

    int trainingStepCounter = 0;
    const int NUM_BATCHES = 10000, batch_size = 16;
    vector<int> seq(input.size());
    for(int i = 0; i < input.size(); i++) seq[i] = i;
    for(int batch = 0; batch < NUM_BATCHES; batch++)
    {
      std::random_shuffle(seq.begin(),seq.end());
      Matrix trainMiniBatch(input[0].size(),batch_size,"input minibatch");
      Matrix targetMiniBatch(target[0].size(),batch_size,"target minibatch");
      for(int i = 0 ; i < batch_size; i++ ){
        for(int j = 0 ; j < input[0].size(); j++ ){
          trainMiniBatch.at(j,i) = input[seq[i]][j];
        }
        cout<<seq[i]<<" ";
      }
      cout<<endl;
      for(int i = 0 ; i < batch_size; i++ ){
        for(int j = 0 ; j < target[0].size(); j++ ){
          targetMiniBatch.at(j,i) = target[seq[i]][j];
        }
      }
      fnn.stepTrain(trainMiniBatch,targetMiniBatch,lr);
      spdlog::info("Training step {} complete", trainingStepCounter);
      trainingStepCounter++;
    }
    
    

    trainDf.close();
    

    ifstream testData;
    testData.open("../data/iris_test.txt");
    vector<string> classNames = {"Iris-setosa","Iris-versicolor","Iris-virginica"};
    spdlog::info("Testing start");
    while(getline(testData, line)){
      vector<string> tokens = split(line, ',');
      string actual = tokens[tokens.size() - 1];
      tokens.pop_back();
      Matrix input(tokens.size(),1,"input");
      for(int i = 0; i < tokens.size(); i++){
        input.at(i, 0) = stof(tokens[i]);
      }
      int _class = fnn.predictClass(input);
      spdlog::info("Prediction {}\tactual: {}\tpredicted: {}", actual == classNames[_class] ? "Correct" : "Wrong", actual, classNames[_class]);
    }
    testData.close();

  } catch (string e) {
    spdlog::error(e);
  }

  return 0;
}
