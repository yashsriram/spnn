#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include <vector>

__global__ void predictForwardPassStep(
    /* Matrix out = ~((~(*in) * weights[i] + biases[i]).sigmoid()); */
    float* out, const int outNR, const int outNC,
    const float* in, const int inNR, const int inNC,
    const float* weightMatrix, const int weightMatrixNR, const int weightMatrixNC,
    const float* biasRowVector, const int biasRowLen) {

    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId >= outNR * outNC) { return; }

    int outRowNum = myId / outNC;
    int outColNum = myId - outRowNum * outNC;

    /* (~*in) * weightMatrix */
    float inTr_MUL_weights = 0;
    for (int k = 0; k < weightMatrixNR; ++k) {
      inTr_MUL_weights += in[k * inNC + outColNum] * weightMatrix[k * weightMatrixNC + outRowNum];
    }

    /* (~*in) * weightMatrix + bias */
    float inTr_MUL_weights_PLUS_bias = inTr_MUL_weights + biasRowVector[outColNum * biasRowLen + outRowNum];

    /* sigmoid((~*in) * weightMatrix + bias) */
    float inTr_MUL_weights_PLUS_bias_SIGMOID = 1.0 / (1.0 + exp(-inTr_MUL_weights_PLUS_bias));

    /* ~(sigmoid((~*in) * weightMatrix + bias)) */
    out[outRowNum * outNC + outColNum] = inTr_MUL_weights_PLUS_bias_SIGMOID;
}

__global__ void fitForwardPassStep(
    /* Matrix out = ~(~(*in) * weights[i] [bias add] biases[i]) */
    /* here bias is a row vector which should be added to all batches */
    float* out, const int outNR, const int outNC,
    const float* in, const int inNR, const int inNC,
    const float* weightMatrix, const int weightMatrixNR, const int weightMatrixNC,
    const float* biasRowVector, const int biasRowLen) {

    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId >= outNR * outNC) { return; }

    int outRowNum = myId / outNC;
    int outColNum = myId - outRowNum * outNC;

    /* (~*in) * weightMatrix */
    float inTr_MUL_weights = 0;
    for (int k = 0; k < weightMatrixNR; ++k) {
      inTr_MUL_weights += in[k * inNC + outColNum] * weightMatrix[k * weightMatrixNC + outRowNum];
    }

    /* (~*in) * weightMatrix [bias add] bias */
    float inTr_MUL_weights_PLUS_bias = inTr_MUL_weights + biasRowVector[outRowNum];

    /* ~((~*in) * weightMatrix + bias) */
    out[outRowNum * outNC + outColNum] = inTr_MUL_weights_PLUS_bias;
}

__global__ void fitBackpropStep(
    /* Matrix temp = ins[i]->sigmoidDerivative() % (weights[i + 1] * *delta) */
    float* out, const int outNR, const int outNC,
    const float* in, const int inNR, const int inNC,
    const float* weightMatrix, const int weightMatrixNR, const int weightMatrixNC,
    const float* delta, const int deltaNR, const int deltaNC) {

    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId >= outNR * outNC) { return; }

    int outRowNum = myId / outNC;
    int outColNum = myId - outRowNum * outNC;

    /* weights[i + 1] * *delta */
    float mulRes = 0;
    for (int k = 0; k < weightMatrixNC; ++k) {
      mulRes += weightMatrix[outRowNum * weightMatrixNC + k] * delta[k * deltaNC + outColNum];
    }

    /* out = ins[i]->sigmoidDerivative() % (weights[i + 1] * *delta) */
    float inVal = in[outRowNum * inNC + outColNum];
    float inValSigmoid = 1.0 / (1.0 + exp(-inVal));
    float inValSigmoidDer = inValSigmoid - inValSigmoid * inValSigmoid;
    out[outRowNum * inNC + outColNum] = inValSigmoidDer * mulRes;

}

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
      printf("Attempt to compile neural net multiple times!\n");
      return;
    }

    int numTrainableParams = 0;
    for(int i = 0; i < layerDims.size() - 1; i++) {
      weights.push_back(Matrix(layerDims[i], layerDims[i + 1]));
      weights[i].setUniform(-1, 1);

      biases.push_back(Matrix(1, layerDims[i + 1]));
      biases[i].setZeros();

      numTrainableParams += weights[i].getNumElements() + biases[i].getNumElements();
      printf("layer: %d\tnumber of nodes: %d\n", i, layerDims[i]);
    }
    printf("layer: %d\tnumber of nodes: %d\n", layerDims.size() - 1, layerDims[layerDims.size() - 1]);
    printf("Total number of trainable parameters : %d\n", numTrainableParams);

    isCompiled = true;
  }

  float crossEntropyLoss(const Matrix& probabilities, const Matrix& target) {
    /* FIXME: possible bug here but not really relavant to computation */
    float totLoss = 0;
    for(int i = 0; i < target.nC; i++){
      std::pair<int, int> trueClassIndex = target.colmax(i);
      if (target.get(trueClassIndex) != 1) {
        std::stringstream ss;
        ss << "Cross Entropy Loss: Prob of true class in target is not 1.\n"
          << "Target is " << target << "\n"
          << "Probabilities is " << probabilities << "\n";
        throw ss.str();
      }
      totLoss += -log(probabilities.get(trueClassIndex));
    }
    return totLoss;
  }

  void fit(const Matrix& inputMatrix, const Matrix& targetMatrix, float learningRate) {
    std::vector<Matrix*> ins,outs;
    Matrix* in = new Matrix(inputMatrix);
    in->name = "Input Matrix";
    Matrix target = Matrix(targetMatrix);
    outs.push_back(in);
    Matrix* out;
    // forward pass
    for(int i = 0; i < weights.size(); i++) {
      /* An example for debugging fitForwardPassStep kernel */
      /* Matrix output(3, 4); */
      /* Matrix input(2, 4); */
      /* for (int j = 0; j < input.nR; ++j) { */
      /*   for (int k = 0; k < input.nC; ++k) { */
      /*     input.set(j, k, j * k); */
      /*   } */
      /* } */
      /* Matrix weightMatrix(2, 3); */
      /* weightMatrix.setUniform(-1, 1); */
      /* Matrix biasRowVector(1, 3); */
      /* biasRowVector.setIdentity(); */
      /* std::cout << ~(~(input) * weightMatrix + biasRowVector) << std::endl; */

      /* fitForwardPassStep<<< (output.nR * output.nC / MAX_THREADS_PER_BLOCK) + 1 , MAX_THREADS_PER_BLOCK >>>( */
      /*     output.getRawPointer(), output.nR, output.nC, */
      /*     input.getConstRawPointer(), input.nR, input.nC, */
      /*     weightMatrix.getConstRawPointer(), weightMatrix.nR, weightMatrix.nC, */
      /*     biasRowVector.getConstRawPointer(), biasRowVector.nC */
      /*     ); */
      /* cudaDeviceSynchronize(); */
      /* std::cout << output << std::endl; */
      /* exit(-1); */

      Matrix temp(weights[i].nC, in->nC);
      fitForwardPassStep<<< (temp.nR * temp.nC / MAX_THREADS_PER_BLOCK) + 1 , MAX_THREADS_PER_BLOCK >>>(
          temp.getRawPointer(), temp.nR, temp.nC,
          in->getConstRawPointer(), in->nR, in->nC,
          weights[i].getConstRawPointer(), weights[i].nR, weights[i].nC,
          biases[i].getConstRawPointer(), biases[i].nC
          );
      cudaDeviceSynchronize();

      in = new Matrix(temp);
      ins.push_back(in);

      out = new Matrix(in->sigmoid());
      outs.push_back(out);

      in = out;
    }

    int i = weights.size() - 1;

    // loss
    /* printf("loss: %f\n", this->crossEntropyLoss(outs[i + 1]->softmax(), target)); */

    // backprop
    Matrix* delta;
    delta = new Matrix((target - outs[i + 1]->softmax()) % ins[i]->sigmoidDerivative());
    Matrix change = (*outs[i]) * (~*delta);
    weights[i] = weights[i] + change * learningRate;

    for(int i = weights.size() - 2 ; i >= 0; i--) {
      /* An example for debugging fitBackpropStep kernel */
      /* Matrix output(2, 3); */
      /* Matrix input(2, 3); */
      /* for (int j = 0; j < input.nR; ++j) { */
      /*   for (int k = 0; k < input.nC; ++k) { */
      /*     input.set(j, k, j * k); */
      /*   } */
      /* } */
      /* Matrix weightMatrix(2, 3); */
      /* weightMatrix.setUniform(-1, 1); */
      /* Matrix del(3, 3); */
      /* del.setIdentity(); */

      /* fitBackpropStep<<< (output.nR * output.nC / MAX_THREADS_PER_BLOCK) + 1 , MAX_THREADS_PER_BLOCK >>>( */
      /*     output.getRawPointer(), output.nR, output.nC, */
      /*     input.getConstRawPointer(), input.nR, input.nC, */
      /*     weightMatrix.getConstRawPointer(), weightMatrix.nR, weightMatrix.nC, */
      /*     del.getConstRawPointer(), del.nR, del.nC */
      /*     ); */
      /* cudaDeviceSynchronize(); */
      /* std::cout <<  input.sigmoidDerivative() % (weightMatrix * del) << std::endl; */
      /* std::cout << output << std::endl; */
      /* exit(-1); */

      Matrix temp(ins[i]->nR, ins[i]->nC);
      fitBackpropStep<<< (temp.nR * temp.nC / MAX_THREADS_PER_BLOCK) + 1 , MAX_THREADS_PER_BLOCK >>>(
          temp.getRawPointer(), temp.nR, temp.nC,
          ins[i]->getConstRawPointer(), ins[i]->nR, ins[i]->nC,
          weights[i + 1].getConstRawPointer(), weights[i + 1].nR, weights[i + 1].nC,
          delta->getConstRawPointer(), delta->nR, delta->nC
          );
      cudaDeviceSynchronize();
      Matrix* delta2 = new Matrix(temp);

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
    delete delta;
  }

  Matrix predict(const Matrix& inputMatrix) const {
    Matrix* in = new Matrix(inputMatrix);
    for(int i = 0; i < weights.size(); i++) {
      /* An example for debugging predictForwardPassStep kernel */
      /* Matrix out(3, 1); */
      /* Matrix input(2, 1); */
      /* input.setOnes(); */
      /* Matrix weightMatrix(2, 3); */
      /* weightMatrix.setUniform(-1, 1); */
      /* Matrix biasRowVector(1, 3); */
      /* biasRowVector.setIdentity(); */
      /* std::cout << input << std::endl; */
      /* std::cout << weightMatrix << std::endl; */
      /* std::cout << biasRowVector << std::endl; */

      /* predictForwardPassStep<<< (out.nR * out.nC / MAX_THREADS_PER_BLOCK) + 1 , MAX_THREADS_PER_BLOCK >>>( */
      /*     out.getRawPointer(), out.nR, out.nC, */
      /*     input.getConstRawPointer(), input.nR, input.nC, */
      /*     weightMatrix.getConstRawPointer(), weightMatrix.nR, weightMatrix.nC, */
      /*     biasRowVector.getConstRawPointer(), biasRowVector.nC */
      /*     ); */
      /* cudaDeviceSynchronize(); */
      /* std::cout << out << std::endl; */
      /* exit(-1); */

      Matrix out(weights[i].nC, in->nC);
      predictForwardPassStep<<< (out.nR * out.nC / MAX_THREADS_PER_BLOCK) + 1 , MAX_THREADS_PER_BLOCK >>>(
          out.getRawPointer(), out.nR, out.nC,
          in->getConstRawPointer(), in->nR, in->nC,
          weights[i].getConstRawPointer(), weights[i].nR, weights[i].nC,
          biases[i].getConstRawPointer(), biases[i].nC
          );
      cudaDeviceSynchronize();

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
