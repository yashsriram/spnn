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


  void step_train(const Matrix& inputMatrix, const Matrix& targetMatrix,float alpha){
    std::vector<Matrix*> ins,outs;
    Matrix* in = new Matrix(inputMatrix);
    in->name = "Input Matrix";
    Matrix target = Matrix(targetMatrix);
    outs.push_back(in);
    Matrix* out;
    // forward pass
    for(int i = 0; i < weights.size(); i++) {
      in->printDims();
      weights[i].printDims();
      in = new Matrix(~(~(*in) * weights[i] + biases[i]));
      out = new Matrix(in->sigmoid());
      in->name =  "Input to layer " + std::to_string(i+1);
      out->name =  "Output of layer " + std::to_string(i+1);
      ins.push_back(in);
      outs.push_back(out);
      in = out;
    }
    spdlog::info("FP Done");
    // backprop
    int n = weights.size();
    Matrix* delta;
    for(int i = n - 1 ; i >= 0; i--){
      if(i == n - 1){
        delta = new Matrix((target - outs[i+1]->softmax()) % *ins[i]);
        delta->name = "delta";
        Matrix change = (*outs[i]) * (~*delta);
        weights[i] = weights[i] + change * alpha;
      } else {
        Matrix* delta2 = new Matrix( (ins[i]->sigmoidDerivative()) % (weights[i+1] * *delta) );
        delete delta;
        delta = delta2;
        Matrix change = (*outs[i]) * (~*delta);
        weights[i] = weights[i] + change * alpha;
      }
    }
    for(auto it : ins){
      delete it;
    }
    for(auto it : outs){
      delete it;
    }
  }


  Matrix forwardPass(const Matrix& inputMatrix) {
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

  int predict(const Matrix& inputMatrix){
    Matrix res = forwardPass(inputMatrix);
    int maxr = -1;
    float maxval = -1000;
    for(int i = 0; i < res.nR; i++){
      if(res.at(i,0) > maxval){
        maxr = i,maxval = res.at(i,0);
      }
    }
    return maxr;
  }

};

#endif
