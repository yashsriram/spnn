#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <spdlog/spdlog.h>
#include <math.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <iomanip>
#include <omp.h>
#include <cblas.h>

bool USE_MATRIX_NAMES = true;

class Matrix {
  std::vector<float> values;
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
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i*nC+j] = m.values[i*nC+j];
      }
    }
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
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i*nC+j] = m.values[i*nC+j];
      }
    }
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

  void set(const int& i, const int& j, const float& k) {
    values[i*nC+j] = k;
  }

  std::pair<int, int> argmax() const {
    auto max_index = std::max_element(values.begin(),values.end()) - values.begin();
    return std::make_pair(max_index/nC, max_index%nC);
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
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i*nC+j] = 0;
      }
    }
    return this;
  }

  Matrix* setOnes() {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i*nC+j] = 1;
      }
    }
    return this;
  }

  Matrix* setIdentity() {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i*nC+j] = (i == j);
      }
    }
    return this;
  }

  Matrix* setUniform(float low, float high) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        float randomNumber = (float) rand() / RAND_MAX;
        randomNumber = low + randomNumber * (high - low);
        values[i*nC+j] = randomNumber;
      }
    }
    return this;
  }

  Matrix sigmoid() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidActivation";
    Matrix result(nR, nC, ss.str());

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i*nC+j] = 1 / (1 + exp(-this->values[i*nC+j]));
      }
    }

    return result;
  }

  Matrix sigmoidDerivative() const {
    std::stringstream ss;
    ss << "(" << name << ")_SigmoidDerivative";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i*nC+j] = 1 / (1 + exp(-this->values[i*nC+j]));
        result.values[i*nC+j] = result.values[i*nC+j] - (result.values[i*nC+j]*result.values[i*nC+j]);
      }
    }

    return result;
  }

  Matrix softmax() const {
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

  Matrix operator~() const {
    std::stringstream ss;
    ss << "(" << name << ")_Transpose";
    Matrix result(nC, nR, USE_MATRIX_NAMES ? ss.str() : "");

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[j*nR+i] = this->values[i*nC+j];
      }
    }

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

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i*nC+j] = this->values[i*nC+j] + m.values[i*nC+j];
      }
    }

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

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i*nC+j] = this->values[i*nC+j] - m.values[i*nC+j];
      }
    }

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

    // omp_set_num_threads(8);

    std::stringstream ss;
    ss << name << " * " << m.name;
    Matrix result(nR, m.nC, USE_MATRIX_NAMES ? ss.str() : "");
    // Matrix result2(nR, m.nC, USE_MATRIX_NAMES ? ss.str() : "");
    
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < m.nC; ++j) {
        float elementSum = 0;
        // #pragma omp parallel for reduction(+:elementSum)
        for (int k = 0; k < nC; ++k) {
          elementSum += this->values[i*nC+k] * m.values[k*m.nC+j];
        }
        result.values[i*m.nC+j] = elementSum;
      }
    }
    // const float* A = values.data();
    // const float* B = m.values.data();
    // float* C = result.values.data();
    // float alpha = 1.f, beta = 0.f;

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         nR, m.nC, nC , alpha, A, nC, B, m.nC, beta, C, m.nC);

    /*for(int i=0; i < result.values.size(); i++){
      if((int(result.values[i]) != int(result2.values[i]))){
        std::cout<<result.values[i]<<" "<<result2.values[i]<<std::endl;
        exit(0);
      }
    }*/
    return result;
  }


  Matrix operator*(float const &value) const {
    std::stringstream ss;
    ss << name << " * " << "const(" << value << ")";
    Matrix result(nR, nC, USE_MATRIX_NAMES ? ss.str() : "");

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i*nC+j] = this->values[i*nC+j] * value;
      }
    }

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

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i*nC+j] = this->values[i*nC+j] * m.values[i*nC+j];
      }
    }

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
