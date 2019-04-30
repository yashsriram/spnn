#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include "../lib/serial/matrix.hpp"

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");

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
    SUM_DEEP_COPY.set(1, 1, 10);
    std::cout << SUM_DEEP_COPY << std::endl;
    std::cout << SUM << std::endl;

    Matrix E(2, 3, "E");
    E.setOnes();
    Matrix MUL = A * E;
    Matrix MUL_CONST = A * 6;
    Matrix MUL_ELEMENTWISE = MUL_CONST % B;
    std::cout << MUL << std::endl;
    std::cout << MUL_CONST << std::endl;
    std::cout << MUL_ELEMENTWISE << std::endl;

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

    Matrix biasSoftmax = bias.softmax();
    std::cout << biasSoftmax << std::endl;

    Matrix output = input * weights + bias;
    std::cout << output << std::endl;

    Matrix F(2, 4, "F");
    F.setIdentity();
    std::cout << F << std::endl;

    Matrix F_sigmoid = F.sigmoid();
    std::cout << F_sigmoid << std::endl;

    Matrix F_softmax = F.softmax();
    std::cout << F_softmax << std::endl;

    Matrix randomized = Matrix(5, 3, "randomized");
    randomized.setUniform(-1, 1);
    std::cout << randomized << std::endl;
    std::pair<int, int> argmax = randomized.argmax();
    spdlog::info("Max of randomized = {} @ ({}, {})", randomized.get(argmax.first, argmax.second), argmax.first, argmax.second);

  } catch (std::string e) {
    std::cerr << e << std::endl;
  }

  return 0;
}
