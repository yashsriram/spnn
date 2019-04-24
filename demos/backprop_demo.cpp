#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "../matrix.hpp"
#include "../nn.hpp"

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

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");

  try {
    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(4);
    fnn.addLayer(10);
    fnn.addLayer(10);
    fnn.addLayer(3);
    fnn.compile();
    float lr = 0.001;

    string line;
    for (int e = 0; e < 1; ++e) {
      ifstream trainData;
      trainData.open("../data/iris_train.txt");
      int trainingStepCounter = 0;
      spdlog::info("Training start");
      while(getline(trainData, line)){
        vector<string> tokens = split(line, ',');
        Matrix target = getTarget(tokens[tokens.size() - 1]);
        tokens.pop_back();
        Matrix input(tokens.size(),1,"input");
        input.setZeros();
        for(int i = 0; i < tokens.size(); i++) {
          input.at(i, 0) = stof(tokens[i]);
        }
        fnn.step_train(input,target,lr);
        spdlog::info("Training step {} complete", trainingStepCounter);
        trainingStepCounter++;
      }
      trainData.close();
    }

    ifstream testData;
    testData.open("../data/iris_test.txt");
    vector<string> outputs = {"Iris-setosa","Iris-versicolor","Iris-virginica"};
    spdlog::info("Testing start");
    while(getline(testData, line)){
      vector<string> tokens = split(line, ',');
      string actual = tokens[tokens.size() - 1];
      tokens.pop_back();
      Matrix input(tokens.size(),1,"input");
      for(int i = 0; i < tokens.size(); i++){
        input.at(i, 0) = stof(tokens[i]);
      }
      spdlog::info("Prediction {}, actual: {}, predicted: {}",
          actual == outputs[fnn.predict(input)] ? "Correct" : "Incorrect", actual, outputs[fnn.predict(input)]);
    }

  } catch (string e) {
    spdlog::error(e);
  }

  return 0;
}
