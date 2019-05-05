#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <math.h>
#include <stdlib.h>
#include <sstream>
#include <algorithm>
#include <utility>
#include <thrust/device_vector.h>

int MAX_THREADS_PER_BLOCK = 1024;
bool USE_MATRIX_NAMES = true;

namespace MatrixKernels {

__global__ void deepCopy(float *a, const float *b, int nR, int nC) {
  a[blockIdx.x * nC + threadIdx.x] = b[blockIdx.x * nC + threadIdx.x];
}

__global__ void setIdentity(float *a, int nR, int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      a[i * nC + j] = (i == j);
    }
  }
}

__global__ void sigmoid(const float *a, float *b, int nR, int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      b[i * nC + j] = 1.0 / (1.0 + exp(-a[i * nC + j]));
    }
  }
}

__global__ void sigmoidDerivative(const float *a, float *b, int nR , int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      float r = 1.0 / (1.0 + exp(-a[i * nC + j]));
      b[i * nC + j] = r - r*r;
    }
  }
}

__global__ void softmax( const float *a, float *b, int nR , int nC) {
  for (int j = 0; j < nC; ++j) {
    float sum = 0;
    for (int i = 0; i < nR; ++i) {
      sum += exp(a[i * nC + j]);
    }
    for (int i = 0; i < nR; ++i) {
      b[i * nC + j] = exp(a[i * nC + j]) / sum;
    }
  }
}

__global__ void transpose(const float *a, float *b, int nR , int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      b[j * nR + i] = a[i * nC + j];
    }
  }
}

__global__ void coladd(const float *a, const float *b, float* c, int nR , int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      c[i * nC + j] = a[i * nC + j] + b[j];
    }
  }
}

__global__ void add( const float *a, const float *b, float* c, int nR , int nC) {
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= nR * nC) {
    return;
  }

  int i = myId / nC;
  int j = myId - i * nC;

  c[i * nC + j] = a[i * nC + j] + b[i * nC + j];
}

__global__ void subtract(const float *a, const float *b, float* c, int nR , int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      c[i * nC + j] = a[i * nC + j] - b[i * nC + j];
    }
  }
}

__global__ void mul(const float *a, const float *b, float* c, int nR , int internalDim, int nC) {
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= nR * nC) {
    return;
  }

  int rowNum = myId / nC;
  int colNum = myId - rowNum * nC;

  c[myId] = 0;
  for (int k = 0; k < internalDim; ++k) {
    c[myId] += a[rowNum * internalDim + k] * b[k * nC + colNum];
  }
}

__global__ void mulScalar( const float *a, const float value, float* c, int nR , int nC) {
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= nR * nC) {
    return;
  }

  int i = myId / nC;
  int j = myId - i * nC;

  c[i * nC + j] = a[i * nC + j] * value;
}

__global__ void mulElementwise( const float *a, const float *b, float* c, int nR , int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      c[i * nC + j] = a[i * nC + j] * b[i * nC + j];
    }
  }
}

};

int DEEP_COPY_COUNTER = 0;

class Matrix {
  thrust::device_vector<float> values;
  friend std::ostream& operator<<(std::ostream&, const Matrix&);
public:
  std::string name;
  const int nR, nC;

  Matrix(int r, int c, std::string name = USE_MATRIX_NAMES ? "<unnamed-matrix>" : ""): nR(r), nC(c), name(name) {
    values.resize(nR*nC);
  }

  Matrix(const Matrix& m) : nR(m.nR), nC(m.nC), name(USE_MATRIX_NAMES ? "(" + m.name + ")_copy" : "") {
    DEEP_COPY_COUNTER++;
    values.resize(nR*nC);
    // deep copy the values variable
    MatrixKernels::deepCopy<<< nR, nC >>>(thrust::raw_pointer_cast(values.data()),
                                     thrust::raw_pointer_cast(m.values.data()), nR, nC);
  }

  void operator=(Matrix const &m) {
    if (m.nC != nC || m.nR != nR) {
      std::stringstream ss;
      ss <<  "Invalid operation: Attempt to assign a matrix with different dimensions: Trying to assign "
         << m.name
         << "(" << m.nR << ", " << m.nC << ")"
         "to matrix"
         << name
         << "(" << nR << ", " << nC << ")";
      throw ss.str();
    }

    name = USE_MATRIX_NAMES ? "(" + m.name + ")_copy" : "";
    MatrixKernels::deepCopy<<< nR, nC >>>(thrust::raw_pointer_cast(values.data()),
                                     thrust::raw_pointer_cast(m.values.data()), nR, nC);
  }

  ~Matrix() { }

  /* Very slow operation. Only use while development in debugging */
  bool operator==(Matrix const &m) {
    if (m.nC != nC || m.nR != nR) {
      return false;
    }

    for (int i = 0; i < values.size(); ++i) {
      if (values[i] != m.values[i]) {
        return false;
      }
    }
    return true;
  }

  int getNumElements() const { return nR * nC; }

  void printDims() const { printf("nR = %d, nC = %d\n", nR, nC); }

  float get(const int& i, const int& j) const {
    return values[i * nC + j];
  }

  float get(const std::pair<int, int>& index) const {
    return this->get(index.first, index.second);
  }

  const float* getConstRawPointer() const {
    return (const float*) thrust::raw_pointer_cast(values.data());
  }

  float* getRawPointer() {
    return thrust::raw_pointer_cast(values.data());
  }

  void set(const int& i, const int& j, const float& k) {
    values[i * nC + j] = k;
  }

  std::pair<int, int> argmax() const {
    auto max_index = std::max_element(values.begin(), values.end()) - values.begin();
    return std::make_pair(max_index / nC, max_index % nC);
  }

  std::pair<int, int> colmax(int col) const {
    float ans = -10000000; // CHANGE IN FUTURE
    int max_ind = -1;
    for (int i = 0; i < nR; ++i) {
      if(values[i*nC+col] > ans){
        max_ind = i;
        ans = values[i*nC+col];
      }
    }
    return std::pair<int, int>(max_ind, col);
  }

  Matrix* setZeros() {
    thrust::fill(values.begin(),values.end(), 0);
    return this;
  }

  Matrix* setOnes() {
    thrust::fill(values.begin(),values.end(), 1);
    return this;
  }

  Matrix* setIdentity() {
    MatrixKernels::setIdentity<<<1, 1>>>(thrust::raw_pointer_cast(values.data()), nR,nC);
    return this;
  }

  Matrix* setUniform(float low, float high) {
    thrust::host_vector<float> randomValues(nR * nC);
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        float randomNumber = (float) rand() / RAND_MAX;
        randomNumber = low + randomNumber * (high - low);
        randomValues[i * nC + j] = randomNumber;
      }
    }
    values = randomValues;
    return this;
  }

  Matrix sigmoid() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidActivation";
    Matrix result(nR, nC, ss.str());

    MatrixKernels::sigmoid<<<1, 1>>>(thrust::raw_pointer_cast(values.data()),
                                     thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }

  Matrix sigmoidDerivative() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidDerivative";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::sigmoidDerivative<<<1, 1>>>(thrust::raw_pointer_cast(values.data()),
                                               thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }

  Matrix softmax() const {
    std::stringstream ss;
    ss << "(" << name << ")_Softmax";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::softmax<<<1, 1>>>(thrust::raw_pointer_cast(values.data()),
                                               thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }

  Matrix operator~() const {
    std::stringstream ss;
    ss << "(" << name << ")_Transpose";
    Matrix result(nC, nR, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::transpose<<<1, 1>>>(thrust::raw_pointer_cast(values.data()),
                                       thrust::raw_pointer_cast(result.values.data()), nR, nC);
    return result;
  }

  Matrix biasAdd(Matrix const &bias) const {
    if (bias.nR == 1 && nC == bias.nC) {
      std::stringstream ss;
      ss << name << " + " << bias.name;
      Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

      MatrixKernels::coladd<<<1, 1>>>(
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(bias.values.data()),
          thrust::raw_pointer_cast(result.values.data()), nR, nC);
      return result;
    }

    std::stringstream ss;
    ss <<  "Invalid dimensions for bias addition: Candidates are matrices "
      << name << "(" << nR << "," << nC << ")"
      << " and "
      << bias.name << "(" << bias.nR << "," << bias.nC << ")";
    throw ss.str();
  }

  Matrix operator+(Matrix const &m) const {
    if (nR != m.nR || nC != m.nC) {
      std::stringstream ss;
      ss <<  "Invalid dimensions for matrix addition: Candidates are matrices "
        << name << "(" << nR << "," << nC << ")"
        << " and "
        << m.name << "(" << m.nR << "," << m.nC << ")";
      throw ss.str();
    }

    std::stringstream ss;
    ss << name << " + " << m.name;
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::add<<<(nR * nC / MAX_THREADS_PER_BLOCK) + 1, MAX_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(m.values.data()),
        thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }

  Matrix operator-(Matrix const &m) const {
    if (nR != m.nR || nC != m.nC) {
      std::stringstream ss;
      ss <<  "Invalid dimensions for matrix subtraction: Candidates are matrices "
        << name << "(" << nR << "," << nC << ")"
        << " and "
        << m.name << "(" << m.nR << "," << m.nC << ")";
      throw ss.str();
    }

    std::stringstream ss;
    ss << name << " - " << m.name;
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::subtract<<<1, 1>>>(
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(m.values.data()),
        thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }

  Matrix operator*(Matrix const &m) const {
    if (nC != m.nR) {
      std::stringstream ss;
      ss <<  "Invalid dimensions for matrix multiplication: Candidates are matrices "
        << name << "(" << nR << "," << nC << ")"
        << " and "
        << m.name << "(" << m.nR << "," << m.nC << ")";
      throw ss.str();
    }

    std::stringstream ss;
    ss << name << " * " << m.name;
    Matrix result(nR, m.nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::mul<<< (nR * m.nC / MAX_THREADS_PER_BLOCK) + 1, MAX_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(m.values.data()),
        thrust::raw_pointer_cast(result.values.data()), nR, nC, m.nC);

    return result;
  }

  Matrix operator*(float const &value) const {
    std::stringstream ss;
    ss << name << " * " << "const(" << value << ")";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::mulScalar<<< nR * nC / MAX_THREADS_PER_BLOCK + 1, MAX_THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(values.data()),
        value,
        thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }


  Matrix operator%(Matrix const &m) const {
    if (nC != m.nC || nR != m.nR) {
      std::stringstream ss;
      ss <<  "Invalid dimensions for matrix element wise multiplication: Candidates are matrices "
        << name << "(" << nR << "," << nC << ")"
        << " and "
        << m.name << "(" << m.nR << "," << m.nC << ")";
      throw ss.str();
    }

    std::stringstream ss;
    ss << name << " % " << m.name;
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::mulElementwise<<<1, 1>>>(
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(m.values.data()),
        thrust::raw_pointer_cast(result.values.data()), nR, nC);

    return result;
  }

};

std::ostream& operator<<(std::ostream &out, const Matrix &m) {
  out << m.name << " of shape: (" << m.nR << "," << m.nC << ") is\n";
  for (int i = 0; i < m.nR; ++i) {
    for (int j = 0; j < m.nC; ++j) {
      out << m.values[i*m.nC+j] << " ";
    }
    out << std::endl;
  }
  return out;
}

#endif
