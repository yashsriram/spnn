#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include "matrix.hpp"

int main() {
  srand(42);
  spdlog::set_pattern("[%^%l%$][%t][%H:%M:%S.%f] %v");

  try {
    Matrix A(3, 2, "A");
    A.setOnes();
    std::cout << A << std::endl;

    Matrix B(3, 2, "B");
    B.setIdentity();
    std::cout << B << std::endl;

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

    Matrix TRANSPOSE = ~B;
    std::cout << TRANSPOSE << std::endl;

    Matrix input(3, 4, "input");
    input.setOnes();
    std::cout << input << std::endl;

    Matrix weights(4, 5, "weights");
    weights.setOnes();
    std::cout << weights << std::endl;

    Matrix bias(3, 5, "bias");
    bias.setIdentity();
    std::cout << bias << std::endl;

    Matrix output = input * weights + bias;
    std::cout << output << std::endl;

  } catch (std::string e) {
    std::cerr << e << std::endl;
  }

  return 0;
}
