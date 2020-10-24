#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <utility>

#include "openmp/matrix.hpp"
#include "openmp/nn.hpp"

using namespace std;

const string TRAIN_FILE_PATH = "data/mnist/train.csv";
const string TEST_FILE_PATH = "data/mnist/test.csv";

const int FEATURES_LEN = 28 * 28;
const int FEATURE_MAX_VALUE = 255;
const int NUM_CLASSES = 10;

const int NUM_EPOCHS = 2;
const int BATCH_SIZE = 16;
const float LEARNING_RATE = 0.01;

vector<string> split(const string& s, char delimiter) {
  vector<string> tokens;
  string token;
  istringstream tokenStream(s);
  while (getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

pair< vector< vector<float> >, vector< vector<int> >  > parseFile(const string& filepath) {
  vector< vector<float> > X;
  vector< vector<int> > y;

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

    vector<int> label(NUM_CLASSES, 0);
    label[stoi(tokens[FEATURES_LEN])] = 1;

    X.push_back(features);
    y.push_back(label);
  }
  stream.close();

  return pair< vector< vector<float> >, vector< vector<int> > >(X, y);
}

int main(int argc, char* argv[]) {
  srand(42);
  USE_MATRIX_NAMES = false;
  // int num_threads = 1;
  // goto_set_num_threads(num_threads);
  // openblas_set_num_threads(num_threads);

  printf("\nConfig:\n\tTRAIN_FILE_PATH: %s\n\tTEST_FILE_PATH: %s\n\tFEATURES_LEN: %d\n\tFEATURE_MAX_VALUE: %d\n\tNUM_CLASSES: %d\n\tNUM_EPOCHS: %d\n\tBATCH_SIZE: %d\n\tLEARNING_RATE: %f\n", TRAIN_FILE_PATH.c_str(), TEST_FILE_PATH.c_str(), FEATURES_LEN, FEATURE_MAX_VALUE, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);

  try {
    vector< vector<float> > train_X;
    vector< vector<int> > train_y;
    auto train_Xy = parseFile(TRAIN_FILE_PATH);
    train_X = train_Xy.first;
    train_y = train_Xy.second;
    printf("Shape of train_X : (%d, %d)\n", train_X.size(), train_X[0].size());
    printf("Shape of train_y : (%d, %d)\n", train_y.size(), train_y[0].size());

    vector< vector<float> > test_X;
    vector< vector<int> > test_y;
    auto test_Xy = parseFile(TEST_FILE_PATH);
    test_X = test_Xy.first;
    test_y = test_Xy.second;
    printf("Shape of test_X  : (%d, %d)\n", test_X.size(), test_X[0].size());
    printf("Shape of test_y  : (%d, %d)\n", test_y.size(), test_y[0].size());

    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(FEATURES_LEN);
    fnn.addLayer(1024);
    fnn.addLayer(512);
    fnn.addLayer(256);
    fnn.addLayer(128);
    fnn.addLayer(64);
    fnn.addLayer(32);
    fnn.addLayer(NUM_CLASSES);
    fnn.compile();

    printf("Training start\n");
    const int NUM_TRAINING_SAMPLES = train_X.size();
    const int NUM_BATCHES = NUM_TRAINING_SAMPLES / BATCH_SIZE + 1;
    /* Mini batch SGD */
    for(int epochNum = 0; epochNum < NUM_EPOCHS; epochNum++) {
      /* shuffling data indices */
      vector<int> seq(NUM_TRAINING_SAMPLES);
      for(int i = 0; i < NUM_TRAINING_SAMPLES; i++) { seq[i] = i; }
      random_shuffle(seq.begin(), seq.end());

      for(int batchNum = 0; batchNum < NUM_BATCHES; batchNum++) {
        Matrix train_X_miniBatch(train_X[0].size(), BATCH_SIZE, "train_X minibatch");
        for(int batch_i = 0 ; batch_i < BATCH_SIZE; batch_i++ ) {
          int randomBatch_i = seq[(batchNum * BATCH_SIZE + batch_i) % NUM_TRAINING_SAMPLES];
          for(int feature_i = 0 ; feature_i < train_X[0].size(); feature_i++ ) {
            train_X_miniBatch.set(feature_i, batch_i, train_X[randomBatch_i][feature_i]);
          }
        }

        Matrix train_y_miniBatch(train_y[0].size(), BATCH_SIZE, "train_y minibatch");
        for(int batch_i = 0 ; batch_i < BATCH_SIZE; batch_i++ ){
          int randomBatch_i = seq[(batchNum * BATCH_SIZE + batch_i) % NUM_TRAINING_SAMPLES];
          for(int feature_i = 0 ; feature_i < train_y[0].size(); feature_i++ ){
            train_y_miniBatch.set(feature_i, batch_i, train_y[randomBatch_i][feature_i]);
          }
        }

        fnn.fit(train_X_miniBatch, train_y_miniBatch, LEARNING_RATE);

        cout << "Epoch : (" << epochNum + 1 << "/" << NUM_EPOCHS << ") Batch: [" << batchNum + 1 << "/" << NUM_BATCHES << "]\r";
        cout.flush();
      }
    }
    cout << "\r\n";

    /* setting up confusion matrix */
    Matrix confusionMatrix(NUM_CLASSES, NUM_CLASSES, "confusion matrix");
    confusionMatrix.setZeros();

    printf("Testing start\n");
    const int NUM_TESTING_SAMPLES = test_X.size();
    for (int testSample_i = 0; testSample_i < NUM_TESTING_SAMPLES; ++testSample_i) {
      vector<float> test_Xi = test_X[testSample_i];
      Matrix testSample(FEATURES_LEN, 1, "testSample");
      for(int j = 0; j < FEATURES_LEN; j++){
        testSample.set(j, 0, test_Xi[j]);
      }

      int actual = -1;
      vector<int> test_yi = test_y[testSample_i];
      for(int j = 0; j < NUM_CLASSES; j++){
        if (test_yi[j] == 1) { actual = j; }
      }

      int prediction = fnn.predictClass(testSample);

      /* printf("Prediction %f\tactual: %f\tpredicted: %f\n", actual == prediction ? "Correct" : "Wrong", actual, prediction); */
      confusionMatrix.set(prediction, actual, confusionMatrix.get(prediction, actual) + 1);

      cout << "Testing : (" << testSample_i + 1 << "/" << NUM_TESTING_SAMPLES << ")\r";
      cout.flush();
    }
    cout << "\r\n";

    /* prediction analysis */
    cout << confusionMatrix;
    int numCorrect = 0;
    int total = NUM_TESTING_SAMPLES;
    for (int i = 0; i < NUM_CLASSES; ++i) {
      numCorrect += confusionMatrix.get(i, i);
    }
    printf("Accuracy: %f\n", (float) numCorrect / total );

  } catch (string e) {
    cout << e << endl;
  }

  return 0;
}
