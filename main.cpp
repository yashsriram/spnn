#include <iostream>
#include "matrix.hpp"

int main() {
  Matrix A = Matrix(3, 2);
  A.setOnes();
  std::cout << A << std::endl;
  return 0;
}
