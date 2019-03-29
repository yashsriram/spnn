#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <sstream>

class Matrix {
  const std::string name;
  const int nR, nC;
  float** values;
  friend std::ostream& operator<<(std::ostream&, const Matrix&);

  void operator=(Matrix const &m) {
    /* = operator is completely disabled to ensure simplicity */
    std::stringstream ss;
    ss <<  "Invalid operation: Attempt to assign a matrix " << m.name.c_str() << " to another matrix";
    throw ss.str();
  }

public:

  Matrix(int r, int c, std::string name = "<unnamed-matrix>"): nR(r), nC(c), name(name) {
    spdlog::info("Matrix {}: constructor called", name.c_str());
    values = new float*[nR];
    for (int i = 0; i < nR; ++i) {
      values[i] = new float[nC];
    }
  }

  Matrix(const Matrix& m) : nR(m.nR), nC(m.nC), name(m.name + "_copy") {
    spdlog::warn("Matrix {}: copy constructor called", name.c_str());
    // allocate heap for values variable
    values = new float*[nR];
    for (int i = 0; i < nR; ++i) {
      values[i] = new float[nC];
    }
    // deep copy the values variable
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i][j] = m.values[i][j];
      }
    }
  }

  ~Matrix() {
    spdlog::info("Matrix {}: destructor is called", name.c_str());
    for (int i = 0; i < nR; ++i) {
      delete[] values[i];
    }
    delete[] values;
  }

  Matrix* setZeros() {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i][j] = 0;
      }
    }
    return this;
  }

  Matrix* setOnes() {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i][j] = 1;
      }
    }
    return this;
  }

  Matrix* setUniform(float low, float high) {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        float randomNumber = (float) rand() / RAND_MAX;
        randomNumber = low + randomNumber * (high - low);
        values[i][j] = randomNumber;
      }
    }
    return this;
  }

  Matrix operator+(Matrix const &m) {
    if (nR != m.nR || nC != m.nC) {
      std::stringstream ss;
      ss <<  "Invalid dimensions for matrix addition: Candidates are matrices " << name << " and " << m.name;
      throw ss.str();
    }

    std::stringstream ss;
    ss << name << " + " << m.name;
    Matrix result(m.nR, m.nC, ss.str());
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i][j] = this->values[i][j] + m.values[i][j];
      }
    }
    return result;
  }

};

std::ostream& operator<<(std::ostream &out, const Matrix &m) {
  for (int i = 0; i < m.nR; ++i) {
    for (int j = 0; j < m.nC; ++j) {
      out << m.values[i][j] << " ";
    }
    out << std::endl;
  }
  return out;
}
