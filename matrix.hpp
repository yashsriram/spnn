#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cublas_v2.h>
#include <thrust/functional.h>

// using namespace thrust::placeholders;

bool USE_MATRIX_NAMES = true;

// /**********************/
// /* cuBLAS ERROR CHECK */
// /**********************/
// #ifndef cublasSafeCall
// #define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
// #endif

// inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
// {
//     if( CUBLAS_STATUS_SUCCESS != err) {
//         fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
//         getch(); cudaDeviceReset(); assert(0); 
//     }
// }

//https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust
struct prg
{
    float a, b;

    __host__ __device__
    prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

struct dev_sigmoid
{
    __host__ __device__
        float operator()(const float x) const
        {
          return (1.0f)/(1+exp(-x));
        }
};

struct dev_sigmoidDerivative
{
    __host__ __device__
        float operator()(const float x) const
        {
          float r = (1.0f)/(1+exp(-x));
          return r - (r*r);
        }
};


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
    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     values[i*nC+j] = m.values[i*nC+j];
    //   }
    // }
    thrust::copy(m.values.begin(),m.values.end(),values.begin());
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
    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     values[i*nC+j] = m.values[i*nC+j];
    //   }
    // }
    thrust::copy(m.values.begin(),m.values.end(),values.begin());
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

  // float& at(const int& i, const int& j) {
  //   return values[i*nC+j];
  // }

  void set(const int& i, const int& j, const float &val) {
    values[i*nC+j] = val;
  }

  void setValues(thrust::host_vector<float> &vec){
    values = vec;
  }


  std::pair<int, int> argmax() const {
    auto max_index = thrust::max_element(values.begin(),values.end()) - values.begin();
    return std::make_pair(max_index/nC, max_index%nC);
  }

  std::pair<int, int> colmax(int col) const {// optimize
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
    thrust::fill(values.begin(),values.end(),0);
    return this;
  }

  Matrix* setOnes() {
    thrust::fill(values.begin(),values.end(),1);
    return this;
  }

  Matrix* setIdentity() { // TO DO
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i*nC+j] = (i == j);
      }
    }
    return this;
  }

  Matrix* setUniform(float low, float high) {
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin,
            index_sequence_begin + nR*nC,
            values.begin(),
            prg(low,high));
    return this;
  }

  Matrix sigmoid() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidActivation";
    Matrix result(nR, nC, ss.str());

    thrust::transform(values.begin(),
            values.end(),
            result.values.begin(),
            dev_sigmoid());

    return result;
  }

  Matrix sigmoidDerivative() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidDerivative";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    thrust::transform(values.begin(),
            values.end(),
            result.values.begin(),
            dev_sigmoidDerivative());

    return result;
  }

  Matrix softmax() const { // TO DO
    std::stringstream ss;
    ss << "(" << name << ")_Softmax";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    for (int j = 0; j < nC; ++j) {
      float sum = 0;
      for (int i = 0; i < nR; ++i) {
        sum += exp(this->values[i*nC+j]);
      }
      for (int i = 0; i < nR; ++i) {
        result.values[i*nC+j] = exp(this->values[i*nC+j]) / sum;
      }
    }

    return result;
  }

  Matrix operator~()  {
    std::stringstream ss;
    ss << "(" << name << ")_Transpose";
    Matrix result(nC, nR, USE_MATRIX_NAMES ? ss.str() : "");

    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     result.values[j*nR+i] = this->values[i*nC+j];
    //   }
    // }
    float* dv_ptr_in  = thrust::raw_pointer_cast(values.data());
    float* dv_ptr_out = thrust::raw_pointer_cast(result.values.data());
    float alpha = 1.;
    float beta  = 0.;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nR, nC, &alpha, dv_ptr_in, nC, &beta, dv_ptr_in, nC, dv_ptr_out, nR); 

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

    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     result.values[i*nC+j] = this->values[i*nC+j] + m.values[i*nC+j];
    //   }
    // }
    const float* dv_ptr_in1  = thrust::raw_pointer_cast(values.data());
    const float* dv_ptr_in2  = thrust::raw_pointer_cast(m.values.data());
    float* dv_ptr_out = thrust::raw_pointer_cast(result.values.data());
    float alpha = 1.;
    float beta  = 1.;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, nC, nR, &alpha, dv_ptr_in1, nC, &beta, dv_ptr_in2, nC, dv_ptr_out, nC); 


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

    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     result.values[i*nC+j] = this->values[i*nC+j] - m.values[i*nC+j];
    //   }
    // }
    const float* dv_ptr_in1  = thrust::raw_pointer_cast(values.data());
    const float* dv_ptr_in2  = thrust::raw_pointer_cast(m.values.data());
    float* dv_ptr_out = thrust::raw_pointer_cast(result.values.data());
    float alpha = 1.;
    float beta  = -1.;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, nC, nR, &alpha, dv_ptr_in1, nC, &beta, dv_ptr_in2, nC, dv_ptr_out, nC); 


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

    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < m.nC; ++j) {
    //     float elementSum = 0;
    //     for (int k = 0; k < nC; ++k) {
    //       elementSum += this->values[i*nC+k] * m.values[k*m.nC+j];
    //     }
    //     result.values[i*m.nC+j] = elementSum;
    //   }
    // }
    const float* dv_ptr_in1  = thrust::raw_pointer_cast(values.data());
    const float* dv_ptr_in2  = thrust::raw_pointer_cast(m.values.data());
    float* dv_ptr_out = thrust::raw_pointer_cast(result.values.data());
    float alpha = 1.;
    float beta  = 0.;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, nR, m.nC, nC, &alpha, dv_ptr_in1, nC, dv_ptr_in2, m.nC, &beta, dv_ptr_out, nR);
    // cudaDeviceSynchronize();
    return result;
  }

  Matrix operator*(float const &value) const {
    std::stringstream ss;
    ss << name << " * " << "const(" << value << ")";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     result.values[i*nC+j] = this->values[i*nC+j] * value;
    //   }
    // }
    thrust::transform(values.begin(), values.end(), result.values.begin(), [=] __host__ __device__ (float x) { return value*x; }); 

    return result;
  }


  Matrix operator%(Matrix const &m) const {
    // if (m.nC == 1 && nR == m.nR) {
    //   std::stringstream ss;
    //   ss << name << " % " << m.name;
    //   Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    //   for (int i = 0; i < nR; ++i) {
    //     for (int j = 0; j < nC; ++j) {
    //       result.values[i][j] = this->values[i][j] * m.values[i][0];
    //     }
    //   }

    //   return result;
    // }

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

    // for (int i = 0; i < nR; ++i) {
    //   for (int j = 0; j < nC; ++j) {
    //     result.values[i*nC+j] = this->values[i*nC+j] * m.values[i*nC+j];
    //   }
    // }
    thrust::transform(values.begin(), values.end(), m.values.begin(), result.values.begin(),thrust::multiplies<float>());

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
