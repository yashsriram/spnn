#include "matrix.hpp"
#include <math.h>

// Activations are all sigmoid
class FullyConnectedNetwork {
  // TODO maintain required weight and bias matrices
  const int inputLayerSize,	outputLayerSize;
  int nLayers = 0;
  vector<int> layers;
  vector<Matrix> weights;
  vector<Matrix> bias;

  public:

  FullyConnectedNetwork(int inputLayer, int outputLayer) {
  	inputLayerSize = inputLayer;
  	outputLayerSize = outputLayer;
  }

  ~FullyConnectedNetwork() {}

  void addLayer(int numberOfNodes) {
    // TODO: just keep track of layer meta data
  	nLayers++;
  	layers.push_back(numberOfNodes);
  }

  void compile() {
    // TODO: using meta data alloc and init required matrices
  	int prevsize = inputLayerSize;
  	for(int i=0; i<=nLayers; i++){
  		if(i==nLayers){
  			weights.push_back(Matrix(prevsize,outputLayerSize));
  			weights[i].setZeros();
  			bias.push_back(Matrix(1,outputLayerSize));
	  		bias[i].setZeros();
  			break;
  		}
  		weights.push_back(Matrix(prevsize,layers[i]));
  		weights[i].setZeros();
  		bias.push_back(Matrix(1,layers[i]));
  		bias[i].setZeros();
  		prevsize = layers[i];
  	}
  }

  float sigmoid(float x)  {
	float exp_value = exp((float) -x);
	return (1 / (1 + exp_value));
  }

  Matrix forwardPass(const Matrix& inputMatrix) {
    // TODO: forward pass
    Matrix prev = inputMatrix;
  	for(int i=0; i<=nLayers; i++){
  		Matrix cur = prev*weights[i];
  		cur = cur.sigmoid(); //add sigmoid to matrix class
  		cur = cur + bias[i];
  		prev = cur;
  	}
  	return prev;
  }

}
