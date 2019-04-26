#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <utility>
#include "../matrix.hpp"
#include "../nn.hpp"

using namespace std;

const int FEATURES_LEN = 28 * 28;
const int FEATURE_MAX_VALUE = 255;
const int NUM_BATCHES = 10;
const int BATCH_SIZE = 16;
const int NUM_EPOCHS = 4000;
const string TRAIN_FILE_PATH = "../data/mnist/train.csv";
const string TEST_FILE_PATH = "../data/mnist/test.csv";

vector<string> split(const string& s, char delimiter) {
  vector<string> tokens;
  string token;
  istringstream tokenStream(s);
  while (getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

pair< vector< vector<float> >, vector< vector<float> >  > parseFile(const string& filepath) {
  vector< vector<float> > X;
  vector< vector<float> > y;

  string line;
  ifstream stream;
  stream.open(filepath);
  while(getline(stream, line)) {
    vector<string> tokens = split(line, ',');
    /* features */
    vector<float> features(FEATURES_LEN, 0);
    for(int i = 0; i < FEATURES_LEN; i++) {
      features[i] = (float) stoi(tokens[i]) / FEATURE_MAX_VALUE;
    }

    vector<float> label(10, 0);
    label[stoi(tokens[FEATURES_LEN])] = 1;

    X.push_back(features);
    y.push_back(label);
  }
  stream.close();

  return pair< vector< vector<float> >, vector< vector<float> > >(X, y);
}

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%^%L%$][%t] %v");
  USE_MATRIX_NAMES = false;

  try {
    vector< vector<float> > train_X;
    vector< vector<float> > train_y;
    auto train_Xy = parseFile(TRAIN_FILE_PATH);
    train_X = train_Xy.first;
    train_y = train_Xy.second;
    spdlog::info("Shape of train_X : ({}, {})", train_X.size(), train_X[0].size());
    spdlog::info("Shape of train_y : ({}, {})", train_y.size(), train_y[0].size());

    vector< vector<float> > test_X;
    vector< vector<float> > test_y;
    auto test_Xy = parseFile(TEST_FILE_PATH);
    test_X = test_Xy.first;
    test_y = test_Xy.second;
    spdlog::info("Shape of test_X  : ({}, {})", test_X.size(), test_X[0].size());
    spdlog::info("Shape of test_y  : ({}, {})", test_y.size(), test_y[0].size());

    exit(0);

    /* auto fnn = FullyConnectedNetwork(); */
    /* fnn.addLayer(4); */
    /* fnn.addLayer(10); */
    /* fnn.addLayer(5); */
    /* fnn.addLayer(3); */
    /* fnn.compile(); */
    /* float lr = 0.0005; */

    /* spdlog::info("Training start"); */
    /* /1* Mini batch SGD *1/ */
    /* vector<int> seq(train_X.size()); */
    /* for(int i = 0; i < train_X.size(); i++) { seq[i] = i; } */
    /* for(int epoch = 0; epoch < NUM_EPOCHS; epoch++) { */
    /*   cout << "Epoch : (" << epoch + 1 << "/" << NUM_EPOCHS << ")\r"; */
    /*   cout.flush(); */
    /*   /1* spdlog::info("Epoch {} complete", epoch); *1/ */
    /*   random_shuffle(seq.begin(),seq.end()); */
    /*   for(int batch = 0; batch < NUM_BATCHES; batch++) { */
    /*     Matrix trainMiniBatch(train_X[0].size(), BATCH_SIZE, "train_X minibatch"); */
    /*     Matrix train_yMiniBatch(train_y[0].size(), BATCH_SIZE, "train_y minibatch"); */
    /*     for(int i = 0 ; i < BATCH_SIZE; i++ ) { */
    /*       for(int j = 0 ; j < train_X[0].size(); j++ ) { */
    /*         trainMiniBatch.at(j,i) = train_X[seq[(batch*BATCH_SIZE + i) % train_X.size()]][j]; */
    /*       } */
    /*     } */
    /*     for(int i = 0 ; i < BATCH_SIZE; i++ ){ */
    /*       for(int j = 0 ; j < train_y[0].size(); j++ ){ */
    /*         train_yMiniBatch.at(j,i) = train_y[seq[(batch*BATCH_SIZE + i) % train_X.size()]][j]; */
    /*       } */
    /*     } */
    /*     fnn.fit(trainMiniBatch, train_yMiniBatch, lr); */
    /*   } */
    /* } */
    /* std::cout << "\r\n"; */

    /* ifstream testData; */
    /* testData.open("../data/iris/test.txt"); */
    /* vector<string> classNames = {"Iris-setosa","Iris-versicolor","Iris-virginica"}; */
    /* spdlog::info("Testing start"); */
    /* while(getline(testData, line)){ */
    /*   vector<string> tokens = split(line, ','); */
    /*   string actual = tokens[tokens.size() - 1]; */
    /*   tokens.pop_back(); */
    /*   Matrix train_X(tokens.size(),1,"train_X"); */
    /*   for(int i = 0; i < tokens.size(); i++){ */
    /*     train_X.at(i, 0) = (stof(tokens[i]) - mins[i])/(maxs[i] - mins[i]); */
    /*   } */
    /*   int _class = fnn.predictClass(train_X); */
    /*   spdlog::info("Prediction {}\tactual: {}\tpredicted: {}", actual == classNames[_class] ? "Correct" : "Wrong", actual, classNames[_class]); */
    /* } */
    /* testData.close(); */

  } catch (string e) {
    spdlog::error(e);
  }

  return 0;
}
