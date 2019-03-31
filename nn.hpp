#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include <vector>

class FullyConnectedNetwork {
  std::vector<int> layerDims;
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
  bool isCompiled;

  public:

  FullyConnectedNetwork() : isCompiled(false) {}

  ~FullyConnectedNetwork() {}

  void addLayer(int layerDim) {
    layerDims.push_back(layerDim);
  }

  void compile() {
    if (isCompiled) {
      spdlog::warn("Attempt to compile neural net multiple times!");
      return;
    }

    int numTrainableParams = 0;
    for(int i = 0; i < layerDims.size() - 1; i++) {
      weights.push_back(Matrix(layerDims[i], layerDims[i + 1]));
      weights[i].setUniform(-1, 1);

      biases.push_back(Matrix(1, layerDims[i + 1]));
      biases[i].setZeros();

      numTrainableParams += weights[i].getNumElements() + biases[i].getNumElements();
    }
    spdlog::info("Total number of trainable parameters : {}", numTrainableParams);

    for(int i = 0; i < weights.size(); i++) {
      std::stringstream name;
      name << "weight_" << i;
      weights[i].setName(name.str());
    }

    for(int i = 0; i < biases.size(); i++) {
      std::stringstream name;
      name << "bias_" << i;
      biases[i].setName(name.str());
    }

    isCompiled = true;
  }

  /* Matrix forwardPass(const Matrix& inputMatrix) { */
  /*   // TODO: forward pass */
  /*   Matrix in = inputMatrix; */
  /*   for(int i=0; i<=nLayers; i++){ */
  /*     Matrix out = (in * weights[i] + biases[i]).sigmoid(); */
  /*     in = out; */
  /*   } */
  /*   return in; */
  /* } */

};

#endif
