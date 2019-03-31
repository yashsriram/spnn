#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <sstream>
#include <vector>

class Matrix {
  const std::string name;
  const int nR, nC;
  std::vector<std::vector<float> > values;
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
    values.resize(nR);
    for (auto& row: values) {
      row.resize(nC);
    }
  }

  Matrix(const Matrix& m) : nR(m.nR), nC(m.nC), name("(" + m.name + ")_copy") {
    spdlog::warn("Matrix {}: copy constructor called", name.c_str());
    values.resize(nR);
    for (auto& row: values) {
      row.resize(nC);
    }
    // deep copy the values variable
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i][j] = m.values[i][j];
      }
    }
  }

  ~Matrix() {
    spdlog::info("Matrix {}: destructor called", name.c_str());
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

  Matrix* setIdentity() {
    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        values[i][j] = (i == j);
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

  float& at(const int& i, const int& j) {
    return values[i][j];
  }

  Matrix operator~() {
    std::stringstream ss;
    ss << "(" << name << ")_Transpose";
    Matrix result(nC, nR, ss.str());

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[j][i] = this->values[i][j];
      }
    }

    return result;
  }

  Matrix operator+(Matrix const &m) {
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
    Matrix result(nR, nC, ss.str());

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i][j] = this->values[i][j] + m.values[i][j];
      }
    }

    return result;
  }

  Matrix operator-(Matrix const &m) {
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
    Matrix result(nR, nC, ss.str());

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < nC; ++j) {
        result.values[i][j] = this->values[i][j] - m.values[i][j];
      }
    }

    return result;
  }

  Matrix operator*(Matrix const &m) {
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
    Matrix result(nR, m.nC, ss.str());

    for (int i = 0; i < nR; ++i) {
      for (int j = 0; j < m.nC; ++j) {
        float elementSum = 0;
        for (int k = 0; k < nC; ++k) {
          elementSum += this->values[i][k] * m.values[k][j];
        }
        result.values[i][j] = elementSum;
      }
    }

    return result;
  }

};

std::ostream& operator<<(std::ostream &out, const Matrix &m) {
  out << m.name << " of shape: (" << m.nR << "," << m.nC << ") is\n";
  for (int i = 0; i < m.nR; ++i) {
    for (int j = 0; j < m.nC; ++j) {
      out << m.values[i][j] << " ";
    }
    out << std::endl;
  }
  return out;
}
