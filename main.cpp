#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include "matrix.hpp"

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");

  try {
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
    spdlog::error(e);
  }

  return 0;
}
