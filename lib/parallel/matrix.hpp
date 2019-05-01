#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <math.h>
#include <stdlib.h>
#include <sstream>
#include <algorithm>
#include <utility>
#include <thrust/device_vector.h>

bool USE_MATRIX_NAMES = true;

namespace MatrixKernels {

__global__ void deepCopy(float *a, const float *b, int nR, int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      a[i * nC + j] = b[i * nC + j];
    }
  }
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
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      c[i * nC + j] = a[i * nC + j] + b[i * nC + j];
    }
  }
}

__global__ void subtract(const float *a, const float *b, float* c, int nR , int nC) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      c[i * nC + j] = a[i * nC + j] - b[i * nC + j];
    }
  }
}

__global__ void mul( const float *a, const float *b, float* c, int nR , int nC, int int_dim) {
  for (int i = 0; i < nR; ++i) {
    for (int j = 0; j < nC; ++j) {
      float elementSum = 0;
      for (int k = 0; k < int_dim; ++k) {
        elementSum += a[i*int_dim+k] * b[k*nC+j];
      }
      c[i * nC + j] = elementSum;
    }
  }
}

__global__ void mulScalar( const float *a, const float value, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i * nC + j] = a[i * nC + j] * value;
      }
    }
}

__global__ void mulElementwise( const float *a, const float *b, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i * nC + j] = a[i * nC + j] * b[i * nC + j];
      }
    }
}

};

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
    values.resize(nR*nC);
    // deep copy the values variable
    MatrixKernels::deepCopy<<<1,1>>>(thrust::raw_pointer_cast(values.data()),
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
    MatrixKernels::deepCopy<<<1,1>>>(thrust::raw_pointer_cast(values.data()),
                                     thrust::raw_pointer_cast(m.values.data()), nR, nC);
  }

  ~Matrix() { }

  int getNumElements() const { return nR * nC; }

  void printDims() const { printf("nR = %d, nC = %d\n", nR, nC); }

  float get(const int& i, const int& j) const {
    return values[i * nC + j];
  }

  float get(const std::pair<int, int>& index) const {
    return this->get(index.first, index.second);
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

  Matrix operator+(Matrix const &m) const {
    if (m.nR == 1 && nC == m.nC) {
      std::stringstream ss;
      ss << name << " + " << m.name;
      Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

      for (int i = 0; i < nR; ++i) {
        for (int j = 0; j < nC; ++j) {
          result.values[i*nC+j] = this->values[i*nC+j] + m.values[j];
        }
      }

      MatrixKernels::coladd<<<1, 1>>>(
          thrust::raw_pointer_cast(values.data()),
          thrust::raw_pointer_cast(m.values.data()),
          thrust::raw_pointer_cast(result.values.data()), nR, nC);
      return result;
    }

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

    MatrixKernels::add<<<1, 1>>>(
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

    MatrixKernels::mul<<<1, 1>>>(
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(m.values.data()),
        thrust::raw_pointer_cast(result.values.data()), nR, m.nC, nC);

    return result;
  }

  Matrix operator*(float const &value) const {
    std::stringstream ss;
    ss << name << " * " << "const(" << value << ")";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    MatrixKernels::mulScalar<<<1, 1>>>(
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
