class Matrix {
  int nR, nC;
  float** values;
  friend std::ostream& operator<<(std::ostream&, const Matrix&);

public:

  Matrix(int r, int c): nR(r), nC(c) {
    values = new float*[nR];
    for (int i = 0; i < nR; ++i) {
      values[i] = new float[nC];
    }
  }

  ~Matrix() {
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
