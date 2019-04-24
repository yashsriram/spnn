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
    if (layerDims.size() <= 1) {
      std::stringstream ss;
      ss <<  "Attempt to compile a neural net with <= 1 layers";
      throw ss.str();
    }

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
      weights[i].name = name.str();
    }

    for(int i = 0; i < biases.size(); i++) {
      std::stringstream name;
      name << "bias_" << i;
      biases[i].name = name.str();
    }

    isCompiled = true;
  }

  float crossEntropyLoss(const Matrix& probabilities, const Matrix& target) {
    std::pair<int, int> trueClassIndex = target.argmax();
    if (target.get(trueClassIndex) != 1) {
      std::stringstream ss;
      ss << "Cross Entropy Loss: Prob of true class in target is not 1.\n"
         << "Target is " << target << "\n"
         << "Probabilities is " << probabilities << "\n";
      throw ss.str();
    }
    return -log(probabilities.get(trueClassIndex));
  }

  void stepTrain(const Matrix& inputMatrix, const Matrix& targetMatrix, float learningRate) {
    std::vector<Matrix*> ins,outs;
    Matrix* in = new Matrix(inputMatrix);
    in->name = "Input Matrix";
    Matrix target = Matrix(targetMatrix);
    outs.push_back(in);
    Matrix* out;
    // forward pass
    for(int i = 0; i < weights.size(); i++) {
      in = new Matrix(~(~(*in) * weights[i] + biases[i]));
      in->name =  "Input to layer " + std::to_string(i+1);
      ins.push_back(in);
      out = new Matrix(in->sigmoid());
      out->name =  "Output of layer " + std::to_string(i+1);
      outs.push_back(out);
      in = out;
    }

    int i = weights.size() - 1;

    // loss
    spdlog::info("loss: {}", this->crossEntropyLoss(outs[i + 1]->softmax(), target));

    // backprop
    Matrix* delta;
    delta = new Matrix((target - outs[i + 1]->softmax()) % *ins[i]);
    delta->name = "delta";
    Matrix change = (*outs[i]) * (~*delta);
    weights[i] = weights[i] + change * learningRate;

    for(int i = weights.size() - 2 ; i >= 0; i--){
      Matrix* delta2 = new Matrix( (ins[i]->sigmoidDerivative()) % (weights[i+1] * *delta) );
      delta->name = "delta2";
      delete delta;
      delta = delta2;
      Matrix change = (*outs[i]) * (~*delta);
      weights[i] = weights[i] + change * learningRate;
    }

    for(auto it : ins){
      delete it;
    }
    for(auto it : outs){
      delete it;
    }
  }

  Matrix predict(const Matrix& inputMatrix) const {
    Matrix* in = new Matrix(inputMatrix);
    for(int i = 0; i < weights.size(); i++) {
      Matrix out = ~((~(*in) * weights[i] + biases[i]).sigmoid());
      delete in;
      in = new Matrix(out);
    }
    Matrix result = *in;
    delete in;
    return result;
  }

  int predictClass(const Matrix& inputMatrix) const {
    Matrix res = predict(inputMatrix);
    std::pair<int, int> resArgmax = res.argmax();
    return resArgmax.first;
  }

};

#endif
