#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include <math.h>
#include <vector>

class FullyConnectedNetwork {
  std::vector<int> layerDims;
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;

  public:

  FullyConnectedNetwork() {}

  ~FullyConnectedNetwork() {}

  void addLayer(int layerDim) {
    layerDims.push_back(layerDim);
  }

  void compile() {
    int numTrainableParams = 0;
    for(int i = 0; i < layerDims.size() - 1; i++) {
      std::stringstream weightsMatrixName;
      weightsMatrixName << "weight_" << i;
      weights.push_back(Matrix(layerDims[i], layerDims[i + 1], weightsMatrixName.str()));
      weights[i].setUniform(-1, 1);

      std::stringstream biasesMatrixName;
      biasesMatrixName << "bias_" << i;
      biases.push_back(Matrix(1, layerDims[i + 1], biasesMatrixName.str()));
      biases[i].setZeros();

      numTrainableParams += weights[i].getNumElements() + biases[i].getNumElements();
    }
    std::cout << "Total number of trainable parameters : " << numTrainableParams << std::endl;
  }

  float sigmoid(float x)  {
    float exp_value = exp((float) -x);
    return (1 / (1 + exp_value));
  }

  /* Matrix forwardPass(const Matrix& inputMatrix) { */
  /*   // TODO: forward pass */
  /*   Matrix prev = inputMatrix; */
  /*   for(int i=0; i<=nLayers; i++){ */
  /*     Matrix cur = prev*weights[i]; */
  /*     cur = cur.sigmoid(); //add sigmoid to matrix class */
  /*     cur = cur + biases[i]; */
  /*     prev = cur; */
  /*   } */
  /*   return prev; */
  /* } */

};

#endif
