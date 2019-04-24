#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "../matrix.hpp"
#include "../nn.hpp"

std::vector<std::string> split(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter))
  {
    tokens.push_back(token);
  }
  return tokens;
}
// https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/

Matrix getTarget(const std::string& s){
  int out_class = -1;

  if(s == "Iris-setosa"){
    out_class = 0;
  }
  else if(s == "Iris-versicolor"){
    out_class = 1;
  }
  else if(s == "Iris-virginica"){
    out_class = 2;
  }
  else{
    throw "unidentified class";
  }
  Matrix result = Matrix(3,1,"target");
  result.setZeros();
  result.at(out_class, 0) = 1;
  return result;
}
std::vector<std::string> outputs = {"Iris-setosa","Iris-versicolor","Iris-virginica"};

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::warn);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");

  try {
    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(4);
    fnn.addLayer(10);
    fnn.addLayer(10);
    fnn.addLayer(3);
    fnn.compile();
    float lr = 0.001;

    std::ifstream datastream;
    datastream.open("../data/iris_train.txt");
    std::string line;
    int trainingStepCounter = 0;
    while(std::getline(datastream,line)){
      std::vector<std::string> tokens = split(line,',');
      Matrix target = getTarget(tokens[tokens.size()-1]);
      tokens.pop_back();
      Matrix input(tokens.size(),1,"input");
      input.setZeros();
      for(int i = 0; i < tokens.size(); i++){
        input.at(i, 0) = std::stof(tokens[i]);
      }
      fnn.step_train(input,target,lr);
      spdlog::critical("Training step {} complete", trainingStepCounter);
      trainingStepCounter++;
    }
    datastream.close();


    std::ifstream testdata;
    testdata.open("../data/iris_test.txt");
    while(std::getline(testdata,line)){
      std::vector<std::string> tokens = split(line,',');
      // Matrix target = getTarget(tokens[tokens.size()-1]);
      std::string actual = tokens[tokens.size()-1];
      tokens.pop_back();
      Matrix input(tokens.size(),1,"input");
      for(int i = 0; i < tokens.size(); i++){
        input.at(i, 0) = std::stof(tokens[i]);
      }
      spdlog::critical("Prediction OK: {}, actual: {}, predicted: {}", actual == outputs[fnn.predict(input)], actual, outputs[fnn.predict(input)]);
    }


  } catch (std::string e) {
    spdlog::error(e);
  }

  return 0;
}
