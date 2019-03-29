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
  B.setIdentity();
  std::cout << B << std::endl;

  try {
    Matrix SUM = A + B;
    std::cout << SUM << std::endl;

    Matrix DIFF = A - B;
    std::cout << DIFF << std::endl;

    Matrix SUM_DEEP_COPY = SUM;
    SUM_DEEP_COPY.at(1, 1) = 10;
    std::cout << SUM_DEEP_COPY << std::endl;
    std::cout << SUM << std::endl;

    Matrix E(2, 3, "E");
    E.setOnes();
    Matrix MUL = A * E;
    std::cout << MUL << std::endl;

  } catch (std::string e) {
    std::cerr << e << std::endl;
  }

  return 0;
}
