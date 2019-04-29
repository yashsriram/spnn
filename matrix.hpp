#ifndef MATRIX_HPP
#define MATRIX_HPP

#define THREADS_PER_BLOCK 1024

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

__global__ void deep_copy( float *a, const float *b) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    a[index] = b[index];
}

__global__ void dev_colmax( const float *a, int* ret, int nR ,int nC, int col, int comps) {
    __shared__ float maxes[1024];
    __shared__ int maxindex[1024];
    float ans = -10000000; // CHANGE IN FUTURE
    int index = threadIdx.x * comps;
    int max_ind = -1;
    for(int i = index; i < index + comps; i++){
      if(i < nR && a[i*nC+col] > ans){
        ans = a[i*nC+col];
        max_ind = i;
      }
    }
    maxes[threadIdx.x] = ans;
    maxindex[threadIdx.x] = max_ind;
    __syncthreads(); 
    if( 0 == threadIdx.x ) {
      int fin = -1;
      float fans = -10000000;
      for( int i = 0; i < blockDim.x; i++ ){
        if(maxes[i] > fans){
          fans = maxes[i];
          fin = maxindex[i];
        }
      }
      *ret = fin;
    }
    
}

__global__ void dev_identity( float *a , int nC) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int xind = index/nC, yind = index%nC;
    if(xind == yind){
      a[index] = 1;
    }
    else{
      a[index] = 0;
    }
}

__global__ void dev_setUniform( float *a, float low, float high) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init((unsigned long long)clock(), 0, 0, &state);
    float rand1 = curand_uniform(&state);
    a[index] = low + rand1 * (high - low);
}

__global__ void dev_sigmoid( const float *a, float *b) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    b[index] = 1.0 / (1.0 + exp(-a[index]));
}

__global__ void dev_sigmoid_derivative( const float *a, float *b) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float r = 1.0 / (1.0 + exp(-a[index]));
    b[index] = r - r*r;
}

__global__ void dev_softmax( const float *a, float *b, int nR , int nC, int nops) {
    int j = blockIdx.x;
    __shared__ float sums[1024];
    __shared__ float totSum;
    int index = threadIdx.x;
    sums[index] = 0;
    for (int i = index*nops; i < index + nops; ++i) {
      if(i < nR) sums[index] += exp(a[i*nC+j]);
    }
    __syncthreads(); 
    if(index == 0){
      totSum = 0;
      for(int i = 0; i < blockDim.x; i++){
        totSum += sums[i];
      }
    }
    __syncthreads();
    for (int i = index*nops; i < index + nops; ++i) {
      if(i < nR) b[i*nC+j] = exp(a[i*nC+j]) / totSum;
    }
}

__global__ void dev_transpose( const float *a, float *b , int nR, int nC) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int xind = index/nC, yind = index%nC;
    b[yind*nR+xind] = a[index];    
}


__global__ void dev_coladd( const float *a, const float *b, float* c , int nC) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int yind = index%nC;
    c[index] = a[index] + b[yind];
}

__global__ void dev_add( const float *a, const float *b, float* c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}


__global__ void dev_sub( const float *a, const float *b, float* c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] - b[index];
}


__global__ void dev_mul( const float *a, const float *b, float* c, int nR , int nC, int int_dim) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int x_ind = index/(nC), y_ind = index%(nC);
    if(x_ind < nR && y_ind < nC){
        int sum = 0;
        for(int i=0; i < int_dim; i++){
            sum += a[x_ind*(int_dim) + i]*b[i*(nC) + y_ind];
        }
        c[index] = sum;
    }
}

__global__ void dev_mulall( const float *a, const float value, float* c, int nR , int nC) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] * value;
}

__global__ void dev_mulelem( const float *a, const float *b, float* c, int nR , int nC) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] * b[index];
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
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    deep_copy<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr);
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
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    deep_copy<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr);
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
    int comparisons_per_thread = 10;
    int num_threads = (nR + comparisons_per_thread - 1)/comparisons_per_thread;
    dev_colmax<<<1,num_threads>>>(val_ptr,dev_mind,nR,nC,col,comparisons_per_thread);
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
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_identity<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,nC);
    return this;
  }

  Matrix* setUniform(float low, float high) {
    float* val_ptr = thrust::raw_pointer_cast(values.data());
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_setUniform<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,low,high);
    return this;
  }

  Matrix sigmoid() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidActivation";
    Matrix result(nR, nC, ss.str());

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_sigmoid<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr);

    return result;
  }

  Matrix sigmoidDerivative() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidDerivative";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_sigmoid_derivative<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr);

    return result;
  }

  Matrix softmax() const {
    std::stringstream ss;
    ss << "(" << name << ")_Softmax";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    // int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    int ops_per_thread = 10;
    dev_softmax<<<nC, (nR+ops_per_thread-1)/ops_per_thread>>>(val_ptr,mval_ptr,nR,nC,ops_per_thread);

    return result;
  }

  Matrix operator~() const {
    std::stringstream ss;
    ss << "(" << name << ")_Transpose";
    Matrix result(nC, nR, USE_MATRIX_NAMES ? ss.str() : "");

    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* mval_ptr = thrust::raw_pointer_cast(result.values.data());
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_transpose<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr,nR,nC);

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
      int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
      dev_coladd<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr,resval_ptr,nC);

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
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_add<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr,resval_ptr);

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
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_sub<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr,resval_ptr);

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
    int num_blocks = (nR*m.nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_mul<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr,resval_ptr,nR,m.nC,nC);    

    return result;
  }

  Matrix operator*(float const &value) const {
    std::stringstream ss;
    ss << name << " * " << "const(" << value << ")";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    
    const float* val_ptr = thrust::raw_pointer_cast(values.data());
    float* resval_ptr = thrust::raw_pointer_cast(result.values.data());
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_mulall<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,value,resval_ptr,nR,nC);    


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
    int num_blocks = (nR*nC + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
    dev_mulelem<<<num_blocks, THREADS_PER_BLOCK>>>(val_ptr,mval_ptr,resval_ptr,nR,nC);

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
