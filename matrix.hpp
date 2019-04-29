#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <spdlog/spdlog.h>
#include <math.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <thrust/device_vector.h>
#include <curand_kernel.h>


bool USE_MATRIX_NAMES = false;

__global__ void deep_copy( float *a, const float *b, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        a[i*nC+j] = b[i*nC+j];
      }
    }
}

__global__ void dev_colmax( const float *a, int* ret, int nR , int nC, int col) {
    float ans = -10000000; // CHANGE IN FUTURE
    for (int i = 0; i < nR; ++i) {
      if(a[i*nC+col] > ans){
        *ret = i;
        ans = a[i*nC+col];
      }
    }
}

__global__ void dev_identity( float *a, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        a[i*nC+j] = (i == j);
      }
    }
}

__global__ void dev_setUniform( float *a, int nR , int nC, float low, float high) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        curandState state;
        curand_init((unsigned long long)clock(), 0, 0, &state);
        float rand1 = curand_uniform(&state);
        a[i*nC+j] = low + rand1 * (high - low);
      }
    }
}

__global__ void dev_sigmoid( const float *a, float *b, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        b[i*nC+j] = 1.0 / (1.0 + exp(-a[i*nC+j]));
      }
    }
}

__global__ void dev_sigmoid_derivative( const float *a, float *b, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        float r = 1.0 / (1.0 + exp(-a[i*nC+j]));
        b[i*nC+j] = r - r*r;
      }
    }
}

__global__ void dev_softmax( const float *a, float *b, int nR , int nC) {
    for (int j = 0; j < nC; ++j) {
      float sum = 0;
      for (int i = 0; i < nR; ++i) {
        sum += exp(a[i*nC+j]);
      }
      for (int i = 0; i < nR; ++i) {
        b[i*nC+j] = exp(a[i*nC+j]) / sum;
      }
    }
}

__global__ void dev_transpose( const float *a, float *b, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        b[j*nR+i] = a[i*nC+j];
      }
    }
}


__global__ void dev_coladd( const float *a, const float *b, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i*nC+j] = a[i*nC+j] + b[j];
      }
    }
}

__global__ void dev_add( const float *a, const float *b, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i*nC+j] = a[i*nC+j] + b[i*nC+j];
      }
    }
}


__global__ void dev_sub( const float *a, const float *b, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i*nC+j] = a[i*nC+j] - b[i*nC+j];
      }
    }
}


__global__ void dev_mul( const float *a, const float *b, float* c, int nR , int nC, int int_dim) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        float elementSum = 0;
        for (int k = 0; k < int_dim; ++k) {
          elementSum += a[i*int_dim+k] * b[k*nC+j];
        }
        c[i*nC+j] = elementSum;
      }
    }
}

__global__ void dev_mulall( const float *a, const float value, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i*nC+j] = a[i*nC+j] * value;
      }
    }
}

__global__ void dev_mulelem( const float *a, const float *b, float* c, int nR , int nC) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        c[i*nC+j] = a[i*nC+j] * b[i*nC+j];
      }
    }
}


class Matrix {
  thrust::device_vector<float> values;
  friend std::ostream& operator<<(std::ostream&, const Matrix&);

public:
  std::string name;
  const int nR, nC;

  Matrix(int r, int c, std::string name = USE_MATRIX_NAMES ? "<unnamed-matrix>" : ""): nR(r), nC(c), name(name) {
    spdlog::debug("Matrix {}: constructor called", name.c_str());
    values.resize(nR*nC);
  }

  Matrix(const Matrix& m) : nR(m.nR), nC(m.nC), name(USE_MATRIX_NAMES ? "(" + m.name + ")_copy" : "") {
    spdlog::debug("Matrix {}: copy constructor called", name.c_str());
    values.resize(nR*nC);
    // deep copy the values variable
    float* val_ptr = thrust::raw_pointer_cast(values.data());
    const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
    deep_copy<<<1,1>>>(val_ptr,mval_ptr,nR,nC);
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
    float* val_ptr = thrust::raw_pointer_cast(values.data());
    const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
    deep_copy<<<1,1>>>(val_ptr,mval_ptr,nR,nC);
  }

  ~Matrix() {
    spdlog::debug("Matrix {}: destructor called", name.c_str());
  }

  int getNumElements() const { return nR * nC; }

  void printDims() const { spdlog::info("nR = {}, nC = {}",nR,nC); }

  float get(const int& i, const int& j) const {
    return values[i*nC+j];
  }

  float get(const std::pair<int, int>& index) const {
    return this->get(index.first, index.second);
  }

  void set(const int& i, const int& j, const float val) {
    values[i*nC+j] = val;
  }
  

  std::pair<int, int> argmax() const { // NOTE SOME PARALLISM IS PRESENT HERE
    auto max_index = thrust::max_element(values.begin(),values.end()) - values.begin();
    return std::make_pair(max_index/nC, max_index%nC);
  }

  std::pair<int, int> colmax(int col) const {
    int* max_ind;
    int* dev_mind;
    max_ind = (int*) malloc(sizeof(int));
    cudaMalloc( (void**) &dev_mind, sizeof(int));
    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    dev_colmax<<<1,1>>>(val_ptr,dev_mind,nR,nC,col);
    cudaMemcpy( max_ind, dev_mind, sizeof(int) , cudaMemcpyDeviceToHost );    
    return std::pair<int, int>(*max_ind, col);
  }

  Matrix* setZeros() {
    thrust::fill(values.begin(),values.end(),0);
    return this;
  }

  Matrix* setOnes() {
    thrust::fill(values.begin(),values.end(),1);
    return this;
  }

  Matrix* setIdentity() {
    float* val_ptr = thrust::raw_pointer_cast(values.data());
    dev_identity<<<1,1>>>(val_ptr,nR,nC);
    return this;
  }

  Matrix* setUniform(float low, float high) {
    float* val_ptr = thrust::raw_pointer_cast(values.data());
    dev_setUniform<<<1,1>>>(val_ptr,nR,nC,low,high);
    return this;
  }

  Matrix sigmoid() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidActivation";
    Matrix result(nR, nC, ss.str());

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_sigmoid<<<1,1>>>(val_ptr,mval_ptr,nR,nC);

    return result;
  }

  Matrix sigmoidDerivative() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidDerivative";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_sigmoid_derivative<<<1,1>>>(val_ptr,mval_ptr,nR,nC);

    return result;
  }

  Matrix softmax() const {
    std::stringstream ss;
    ss << "(" << name << ")_Softmax";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_softmax<<<1,1>>>(val_ptr,mval_ptr,nR,nC);

    return result;
  }

  Matrix operator~() const {
    std::stringstream ss;
    ss << "(" << name << ")_Transpose";
    Matrix result(nC, nR, USE_MATRIX_NAMES ? ss.str() : "");

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_transpose<<<1,1>>>(val_ptr,mval_ptr,nR,nC);

    return result;
  }

  Matrix operator+(Matrix const &m) const {
    if (m.nR == 1 && nC == m.nC) {
      std::stringstream ss;
      ss << name << " + " << m.name;
      Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

      const float* val_ptr = thrust::raw_pointer_cast(values.data());
      const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
      float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
      dev_coladd<<<1,1>>>(val_ptr,mval_ptr,resval_ptr,nR,nC);

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

    
    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
    float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_add<<<1,1>>>(val_ptr,mval_ptr,resval_ptr,nR,nC);

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

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
    float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_sub<<<1,1>>>(val_ptr,mval_ptr,resval_ptr,nR,nC);

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

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
    float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_mul<<<1,1>>>(val_ptr,mval_ptr,resval_ptr,nR,m.nC,nC);    

    return result;
  }

  Matrix operator*(float const &value) const {
    std::stringstream ss;
    ss << name << " * " << "const(" << value << ")";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    
    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_mulall<<<1,1>>>(val_ptr,value,resval_ptr,nR,nC);    


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

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    const float* mval_ptr = thrust::raw_pointer_cast(m.values.data());
    float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
    dev_mulelem<<<1,1>>>(val_ptr,mval_ptr,resval_ptr,nR,nC);

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
