#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include "../lib/cuda/matrix.hpp"
#include "../lib/cuda/nn.hpp"

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
  result.set(_class, 0, 1);
  return result;
}

vector<float> getTargetVector(const std::string& s){
  int _class;
  vector<float> result(3, 0);
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
  srand(42);
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("%v");
  USE_MATRIX_NAMES = false;

  try {
    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(4);
    fnn.addLayer(10);
    fnn.addLayer(5);
    fnn.addLayer(3);
    fnn.compile();
    float lr = 0.0005;

    string line;
    spdlog::info("Training start");

    ifstream trainDf;
    trainDf.open("../data/iris/train.txt");
    vector< vector<float> > input;
    vector< vector<float> > target;
    while(getline(trainDf, line)) {
      vector<string> tokens = split(line, ',');
      vector<float> tgt = getTargetVector(tokens[tokens.size() - 1]);
      tokens.pop_back();
      vector<float> inp(tokens.size(), 0);
      for(int i = 0; i < tokens.size(); i++) {
        inp[i] = stof(tokens[i]);
      }
      input.push_back(inp);
      target.push_back(tgt);
    }

    /* Normalization */
    vector<float> mins(input[0].size(),1000000),maxs(input[0].size(),0) ;
    for(int i = 0; i < input.size(); i++) {
      for(int j = 0; j < input[0].size(); j++) {
        mins[j] = min(mins[j],input[i][j]);
        maxs[j] = max(maxs[j],input[i][j]);
      }
    }
    for(int i = 0; i < input.size(); i++) {
      for(int j = 0; j < input[0].size(); j++) {
        input[i][j] = (input[i][j] - mins[j])/(maxs[j] - mins[j]);
      }
    }

    /* Mini batch SGD */
    const int NUM_BATCHES = 10;
    const int BATCH_SIZE = 16;
    const int NUM_EPOCHS = 4000;
    vector<int> seq(input.size());
    for(int i = 0; i < input.size(); i++) { seq[i] = i; }
    for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
      spdlog::info("Epoch {} complete", epoch);
      random_shuffle(seq.begin(),seq.end());
      for(int batch = 0; batch < NUM_BATCHES; batch++) {
        Matrix trainMiniBatch(input[0].size(), BATCH_SIZE, "input minibatch");
        Matrix targetMiniBatch(target[0].size(), BATCH_SIZE, "target minibatch");
        for(int i = 0 ; i < BATCH_SIZE; i++ ) {
          for(int j = 0 ; j < input[0].size(); j++ ) {
            trainMiniBatch.set(j,i,input[seq[(batch*BATCH_SIZE + i) % input.size()]][j]);
          }
        }
        for(int i = 0 ; i < BATCH_SIZE; i++ ){
          for(int j = 0 ; j < target[0].size(); j++ ){
            targetMiniBatch.set(j,i,target[seq[(batch*BATCH_SIZE + i) % input.size()]][j]);
          }
        }
        fnn.fit(trainMiniBatch, targetMiniBatch, lr);
      }
    }

    trainDf.close();

    ifstream testData;
    testData.open("../data/iris/test.txt");
    vector<string> classNames = {"Iris-setosa","Iris-versicolor","Iris-virginica"};
    spdlog::info("Testing start");
    while(getline(testData, line)){
      vector<string> tokens = split(line, ',');
      string actual = tokens[tokens.size() - 1];
      tokens.pop_back();
      Matrix input(tokens.size(),1,"input");
      for(int i = 0; i < tokens.size(); i++){
        input.set(i, 0, (stof(tokens[i]) - mins[i])/(maxs[i] - mins[i]));
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
