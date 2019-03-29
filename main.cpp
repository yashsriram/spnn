#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include "matrix.hpp"

int main() {
  srand(42);
  spdlog::set_pattern("[%^%l%$][%t][%H:%M:%S.%f] %v");

  Matrix A(3, 2, "A");
  A.setOnes();
  std::cout << A << std::endl;

  Matrix B(3, 2, "B");
  B.setOnes();
  std::cout << B << std::endl;

  try {
    Matrix C = A + B;
    std::cout << C << std::endl;
    Matrix D = A;
  } catch (std::string e) {
    std::cerr << e << std::endl;
  }

  return 0;
}
